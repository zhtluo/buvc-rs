use crate::{
    poly::{compute_unity, poly_delta, poly_fft, poly_ifft, poly_interpolate, poly_zero},
    vc_parameter::VcParameter,
};
use ark_bls12_381::{fr::Fr, Bls12_381, G1Projective as G1, G2Projective as G2};
use ark_ec::{pairing::Pairing, Group};
use ark_ff::{
    fields::{FftField, Field},
    Zero,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use std::iter;

/// Precomputed context in VC
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct VcContext {
    pub n: usize,
    pub logn: usize,
    /// n as a field element
    pub nf: Fr,
    /// Root of unity w^0, w^1, ..., w^(n-1)
    pub unity: Vec<Fr>,
    /// Group element L_0, L_1, ..., L_(n-1), as described in the paper
    pub gl: Vec<G1>,
    /// Group element L'_0, L'_1, ..., L'_(n-1), as described in the paper
    pub gll: Vec<G1>,
}

impl VcContext {
    pub fn new(vc_p: &VcParameter, logn: usize) -> VcContext {
        let n = 1 << logn;
        let gs = &vc_p.gs1;

        assert!(n <= vc_p.n);
        assert!(logn <= Fr::TWO_ADICITY as usize);
        let unity: Vec<Fr> = compute_unity(n);

        let mut gl: Vec<G1> = (0..n).map(|i| gs[n - i - 1]).collect();
        poly_fft(&unity, &mut gl, n);

        let mut gll: Vec<G1> = (0..n - 1)
            .map(|i| gs[n - i - 2] * Fr::from((i + 1) as u32))
            .chain(iter::once(G1::zero()))
            .collect();
        poly_fft(&unity, &mut gll, n);

        return VcContext {
            n,
            logn,
            nf: Fr::from(n as u32),
            unity,
            gl,
            gll,
        };
    }

    /// Single-point verification
    pub fn verify(&self, vc_p: &VcParameter, gc: G1, index: usize, value: Fr, gq: G1) -> bool {
        let n = self.n;
        let unity = &self.unity;
        let gs2 = &vc_p.gs2;
        let step = unity.len() / n;

        let lhs = Bls12_381::pairing(gq, gs2[1] - G2::generator() * unity[index * step]);
        let rhs = Bls12_381::pairing(gc - G1::generator() * value, G2::generator());

        lhs == rhs
    }

    /// Multi-point verification
    pub fn verify_multi(
        &self,
        vc_p: &VcParameter,
        gc: G1,
        index: &[usize],
        value: &[Fr],
        gq: G1,
    ) -> bool {
        let n = self.n;
        let unity = &self.unity;
        let gs1 = &vc_p.gs1;
        let gs2 = &vc_p.gs2;
        let step = unity.len() / n;

        let x = &index.iter().map(|i| unity[*i * step]).collect::<Vec<Fr>>();
        let z = poly_zero(unity, &x);
        let lag = poly_interpolate(unity, &x, value);

        let lhs = Bls12_381::pairing(gq, (0..z.len()).map(|i| gs2[i] * z[i]).sum::<G2>());
        let rhs = Bls12_381::pairing(
            gc - (0..lag.len()).map(|i| gs1[i] * lag[i]).sum::<G1>(),
            G2::generator(),
        );

        lhs == rhs
    }

    /// Build a commitment and witnesses from v
    pub fn build_commitment(&self, v: &[Fr]) -> (G1, Vec<G1>) {
        let n = v.len();
        assert!(self.n == n);

        let nf = self.nf;
        let unity = &self.unity;
        let gl = &self.gl;
        let gll = &self.gll;
        let step = unity.len() / n;

        let mut a: Vec<Fr> = (0..n).map(|i| v[i] * unity[i * step] / nf).collect();
        let mut b: Vec<G1> = (0..n)
            .map(|i| gl[i] * (v[i] * unity[i * step] / nf))
            .collect();
        let gc = b.iter().sum();
        poly_fft(unity, &mut a, n);
        poly_fft(unity, &mut b, n);
        let tau: Vec<Fr> = (0..n)
            .map(|i| (nf - Fr::from((i * 2 + 1) as u32)) / Fr::from(2))
            .collect();
        a = a.into_iter().zip(&tau).map(|(i, j)| i * j).collect();
        b = b.into_iter().zip(&tau).map(|(i, j)| i * j).collect();
        poly_ifft(unity, &mut a, n);
        poly_ifft(unity, &mut b, n);
        a = a
            .into_iter()
            .zip(unity.iter().step_by(step))
            .map(|(i, j)| i * (Fr::ONE / j))
            .collect();
        b = b
            .into_iter()
            .zip(unity.iter().step_by(step))
            .map(|(i, j)| i * (Fr::ONE / j))
            .collect();

        let gq: Vec<G1> = (0..n)
            .map(|i| gll[i] * (v[i] * unity[i * step] / nf) + gl[i] * a[i] - b[i])
            .collect();

        (gc, gq)
    }

    /// Produce a multi-proof given a list of proofs
    pub fn aggregate_proof(&self, index: &[usize], gq: &[G1]) -> G1 {
        let n = self.n;
        let unity = &self.unity;
        let step = unity.len() / n;
        let x = &index.iter().map(|i| unity[*i * step]).collect::<Vec<Fr>>();
        let d = poly_delta(unity, &x);
        (0..index.len()).map(|i| gq[i] * (Fr::ONE / d[i])).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::vc_parameter::tests::test_parameter;
    use ark_ff::Field;

    #[test]
    fn test_vc_context_new() {
        let (_s, vc_p) = test_parameter();
        let vc_c = VcContext::new(&vc_p, vc_p.logn);

        let n = vc_c.n;
        let gs1 = &vc_p.gs1;
        let unity = &vc_c.unity;
        let step = unity.len() / n;

        let mut gl: Vec<G1> = Vec::with_capacity(n);
        for i in 0..n {
            let mut x_power = Fr::ONE;
            gl.push(G1::zero());
            for j in 0..n {
                gl[i] = gl[i] + gs1[n - j - 1] * x_power;
                x_power = x_power * unity[i * step];
            }
        }

        assert_eq!(gl, vc_c.gl);

        let mut gll: Vec<G1> = Vec::with_capacity(n);
        for i in 0..n {
            let mut x_power = Fr::ONE;
            gll.push(G1::zero());
            for j in 0..n - 1 {
                gll[i] = gll[i] + gs1[n - j - 2] * (Fr::from((j + 1) as u32) * x_power);
                x_power = x_power * unity[i * step];
            }
        }

        assert_eq!(gll, vc_c.gll);
    }

    #[test]
    fn test_vc_context_commitment() {
        let (s, vc_p) = test_parameter();
        let vc_c = VcContext::new(&vc_p, 1);
        let n = vc_c.n;
        let v = [0, 1].map(|i| Fr::from(i));
        let unity = &vc_c.unity;
        let step = unity.len() / n;

        let (gc, gq) = vc_c.build_commitment(&v);
        assert_eq!(
            gc,
            G1::generator()
                * (s * (Fr::ONE / (unity[step] - unity[0])) - (Fr::ONE / (unity[step] - unity[0])))
        );
        assert_eq!(
            gq[0],
            G1::generator() * (Fr::ONE / (unity[step] - unity[0]))
        );
        assert!(vc_c.verify(&vc_p, gc, 0, v[0], gq[0]));
        assert!(vc_c.verify(&vc_p, gc, 1, v[1], gq[1]));
    }

    #[test]
    fn test_vc_context_verify() {
        let (_s, vc_p) = test_parameter();
        let vc_c = VcContext::new(&vc_p, vc_p.logn);
        let v = [1, 4, 5, 2, 3, 6, 7, 0].map(|i| Fr::from(i));

        let n = vc_c.n;
        let (gc, gq) = vc_c.build_commitment(&v);
        for i in 0..n {
            assert!(vc_c.verify(&vc_p, gc, i, v[i], gq[i]));
        }
        for i in 0..n {
            assert!(!vc_c.verify(&vc_p, gc, i, v[i] + Fr::ONE, gq[i]));
        }
    }

    #[test]
    fn test_vc_context_verify_multi() {
        let (_s, vc_p) = test_parameter();
        let vc_c = VcContext::new(&vc_p, vc_p.logn);
        let v = [1, 4, 5, 2, 3, 6, 7, 0].map(|i| Fr::from(i));

        let (gc, gq) = vc_c.build_commitment(&v);
        let gqq = vc_c.aggregate_proof(&[0, 1, 2], &gq[0..=2]);
        assert!(vc_c.verify_multi(&vc_p, gc, &[0, 1, 2], &v[0..=2], gqq));
        
        let (gc, gq) = vc_c.build_commitment(&v);
        let gqq = vc_c.aggregate_proof(&[1, 2, 3], &gq[1..=3]);
        assert!(vc_c.verify_multi(&vc_p, gc, &[1, 2, 3], &v[1..=3], gqq));
        
        let (gc, gq) = vc_c.build_commitment(&v);
        let gqq = vc_c.aggregate_proof(&[1, 2, 3, 4], &gq[1..=4]);
        assert!(vc_c.verify_multi(&vc_p, gc, &[1, 2, 3, 4], &v[1..=4], gqq));

        let index = [3, 4, 5, 6];
        let gqq = vc_c.aggregate_proof(&index, &gq[3..=6]);
        assert!(vc_c.verify_multi(&vc_p, gc, &index, &v[3..=6], gqq));

        let index = [0, 1, 2, 3, 4, 5, 6];
        let gqq = vc_c.aggregate_proof(&index, &gq[0..=6]);
        assert!(vc_c.verify_multi(&vc_p, gc, &index, &v[0..=6], gqq));
    }
}
