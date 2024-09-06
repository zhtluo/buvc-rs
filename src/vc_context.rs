use std::iter;
use std::cmp::Ordering; // Import Ordering for comparison

use ark_bls12_381::fr::Fr;
use ark_bls12_381::Bls12_381;
use ark_bls12_381::G1Projective as G1;
use ark_bls12_381::G2Projective as G2;
use ark_ec::pairing::Pairing;
use ark_ec::Group;
use ark_ff::fields::FftField;
use ark_ff::fields::Field;
use ark_ff::Zero;
use ark_serialize::CanonicalDeserialize;
use ark_serialize::CanonicalSerialize;
use ark_std::iterable::Iterable;

use crate::poly::compute_unity;
use crate::poly::poly_delta;
use crate::poly::poly_evaluate;
use crate::poly::poly_fft;
use crate::poly::poly_ifft;
use crate::poly::poly_interpolate;
use crate::poly::poly_zero;
use crate::vc_parameter::VcParameter;

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

        VcContext {
            n,
            logn,
            nf: Fr::from(n as u32),
            unity,
            gl,
            gll,
        }
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
        let z = poly_zero(unity, x);
        let lag = poly_interpolate(unity, x, value);

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

    /// Update commitment with a new delta
    pub fn update_commitment(&self, gc: G1, index: usize, value: Fr) -> G1 {
        let n = self.n;
        let nf = self.nf;
        let unity = &self.unity;
        let step = unity.len() / n;
        let gl = &self.gl;

        gc + gl[index] * (value * unity[index * step] / nf)
    }

    pub fn update_witness_batchpub(
        &self,
        alpha: &[usize], // Indices of the proofs
        gq: &[G1],       // The G1 elements corresponding to the proofs
        beta: &[usize],  // Indices of the modifications
        value: &[Fr],    // The Fr elements corresponding to the modifications
    ) -> () {
    
        // Sorting the indices if necessary
        let mut sorted_alpha = alpha.to_vec();
        let mut sorted_beta = beta.to_vec();
        sorted_alpha.sort_unstable();
        sorted_beta.sort_unstable();
    
        // Partition into matching and differing indices
        let mut common_alpha = Vec::new();  // For matching alpha indices
        let mut common_beta = Vec::new();   // For matching beta indices
        let mut alpha_extra = Vec::new();   // For extra alpha indices
        let mut beta_extra = Vec::new();    // For extra beta indices
    
        let mut i = 0;
        let mut j = 0;
    
        // Finding matching and extra indices
        while i < sorted_alpha.len() && j < sorted_beta.len() {
            match sorted_alpha[i].cmp(&sorted_beta[j]) {
                Ordering::Equal => {
                    common_alpha.push(sorted_alpha[i]);
                    common_beta.push(sorted_beta[j]);
                    i += 1;
                    j += 1;
                }
                Ordering::Less => {
                    alpha_extra.push(sorted_alpha[i]);
                    i += 1;
                }
                Ordering::Greater => {
                    beta_extra.push(sorted_beta[j]);
                    j += 1;
                }
            }
        }
    
        // Add remaining elements
        while i < sorted_alpha.len() {
            alpha_extra.push(sorted_alpha[i]);
            i += 1;
        }
        while j < sorted_beta.len() {
            beta_extra.push(sorted_beta[j]);
            j += 1;
        }
    
        // Handle the different cases
        
        self.update_witness_batch_equal(&common_alpha, gq, &common_beta, value);
        
        self.update_witnesses_batch_different(&common_alpha, gq, &beta_extra, value);
        
        self.update_witnesses_batch_different(&alpha_extra, gq, beta, value);
            
    }

    
    pub fn update_witness_batch_equal(
        &self,
        alpha: &[usize],
        gq: &[G1],
        beta: &[usize],
        value: &[Fr],
    ) -> () {
        println!("Equal");

    }

    /// Update witnesses given updates (\alpha, \beta)
    /// elements in \alpha must be different from index
    /// Test TBD
    pub fn update_witnesses_batch_different(
        &self,
        alpha: &[usize],
        gq: &[G1],
        beta: &[usize],
        value: &[Fr],
    ) -> Vec<G1> {
        let nf = &self.nf;
        let unity = &self.unity;

        let walpha = alpha.iter().map(|i| unity[*i]).collect::<Vec<Fr>>();
        let wbeta = beta.iter().map(|i| unity[*i]).collect::<Vec<Fr>>();
        let z = poly_zero(unity, &walpha);

        let zz = z
            .iter()
            .skip(1)
            .enumerate()
            .map(|(i, x)| x * &Fr::from((i + 1) as u32))
            .collect::<Vec<Fr>>();

        let zwindex = poly_evaluate(unity, &z, &wbeta);
        let zzwalpha = poly_evaluate(unity, &zz, &walpha);

        let ya: Vec<Fr> = value
            .iter()
            .zip(walpha.iter().zip(zzwalpha.iter()))
            .map(|(v, (u, a))| v * u / nf * a)
            .collect();
        let yb: Vec<G1> = ya
            .iter()
            .enumerate()
            .map(|(i, v)| self.gl[alpha[i]] * v)
            .collect();
        let ha = poly_interpolate(&self.unity, &walpha, &ya);
        let hb = poly_interpolate(&self.unity, &walpha, &yb);

        let evala = poly_evaluate(&self.unity, &ha, &wbeta);
        let evalb = poly_evaluate(&self.unity, &hb, &wbeta);

        let lindex = beta.iter().map(|i| self.gl[*i]);
        gq.iter()
            .zip(evala.iter().zip(evalb.iter()))
            .zip(zwindex.iter().zip(lindex))
            .map(|((g, (a, b)), (z, l))| g + l * (a / z) - *b * (Fr::ONE / z))
            .collect()
    }

    /// Produce a multi-proof given a list of proofs
    pub fn aggregate_proof(&self, index: &[usize], gq: &[G1]) -> G1 {
        let n = self.n;
        let unity = &self.unity;
        let step = unity.len() / n;
        let x = &index.iter().map(|i| unity[*i * step]).collect::<Vec<Fr>>();
        let d = poly_delta(unity, x);
        (0..index.len()).map(|i| gq[i] * (Fr::ONE / d[i])).sum()
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::Field;

    use super::*;
    use crate::vc_parameter::tests::test_parameter;

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
                gl[i] += gs1[n - j - 1] * x_power;
                x_power *= unity[i * step];
            }
        }

        assert_eq!(gl, vc_c.gl);

        let mut gll: Vec<G1> = Vec::with_capacity(n);
        for i in 0..n {
            let mut x_power = Fr::ONE;
            gll.push(G1::zero());
            for j in 0..n - 1 {
                gll[i] += gs1[n - j - 2] * (Fr::from((j + 1) as u32) * x_power);
                x_power *= unity[i * step];
            }
        }

        assert_eq!(gll, vc_c.gll);
    }

    #[test]
    fn test_vc_context_commitment() {
        let (s, vc_p) = test_parameter();
        let vc_c = VcContext::new(&vc_p, 1);
        let n = vc_c.n;
        let v = [0, 1].map(Fr::from);
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
        let v = [1, 4, 5, 2, 3, 6, 7, 0].map(Fr::from);

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
        let v = [1, 4, 5, 2, 3, 6, 7, 0].map(Fr::from);

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

    //Test for update_witness_batch
    #[test]
    fn test_update_witness_batchpub() {
        

        // Further assertions can be added depending on the behavior you want to test
        // such as checking which cases were executed, if the result is correct, etc.
    }
}
