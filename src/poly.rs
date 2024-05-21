/// This module handles all polynomial operations.
///
use ark_bls12_381::fr::Fr;
use ark_ff::{FftField, Field};
use std::{
    cmp::max,
    iter,
    ops::{Add, Mul, Sub},
};

/// Compute all unity of roots for n
pub fn compute_unity(n: usize) -> Vec<Fr> {
    let n = n.next_power_of_two();
    let u: Fr = Fr::get_root_of_unity(n as u64).unwrap();
    iter::successors(Some(Fr::ONE), |&prev| Some(prev * u))
        .take(n)
        .collect()
}

/// Trim additional zeroes at the end of the polynomial.
pub fn trim_zeroes(p: &mut Vec<Fr>) {
    while !p.is_empty() && p[p.len() - 1] == Fr::ZERO {
        p.pop();
    }
}

/// Compute FFT on a, where a can be any vector that supports scalar multiplication with Fr.
pub fn poly_fft<T>(unity: &[Fr], a: &mut Vec<T>, n: usize)
where
    T: Add<T, Output = T> + Sub<T, Output = T> + Mul<Fr, Output = T> + Default + Copy,
{
    let n = n.next_power_of_two();
    assert!(n <= unity.len());
    a.resize_with(n, Default::default);
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j >= bit {
            j -= bit;
            bit >>= 1;
        }
        j += bit;

        if i < j {
            a.swap(i, j);
        }
    }

    let mut len = 2;
    while len <= n {
        let half_len = len / 2;
        let step = unity.len() / len;
        for start in (0..n).step_by(len) {
            for k in 0..half_len {
                let u = a[start + k];
                let t = a[start + k + half_len] * unity[k * step];
                a[start + k] = u + t;
                a[start + k + half_len] = u - t;
            }
        }
        len *= 2;
    }
}

/// Compute iFFT on a, where a can be any vector that supports scalar multiplication with Fr.
pub fn poly_ifft<T>(unity: &[Fr], a: &mut Vec<T>, n: usize)
where
    T: Add<T, Output = T> + Sub<T, Output = T> + Mul<Fr, Output = T> + Default + Copy,
{
    let n = n.next_power_of_two();
    assert!(n <= unity.len());
    a.resize_with(n, Default::default);
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j >= bit {
            j -= bit;
            bit >>= 1;
        }
        j += bit;

        if i < j {
            a.swap(i, j);
        }
    }

    let mut len = 2;
    while len <= n {
        let half_len = len / 2;
        let step = unity.len() / len;
        for start in (0..n).step_by(len) {
            for k in 0..half_len {
                let u = a[start + k];
                let t = a[start + k + half_len] * (Fr::ONE / unity[k * step]);
                a[start + k] = u + t;
                a[start + k + half_len] = u - t;
            }
        }
        len *= 2;
    }

    let inv_n = Fr::ONE / Fr::from(n as u32);
    for i in 0..n {
        a[i] = a[i] * inv_n;
    }
}

/// Compute the zeroing polynomial in O(mlog^2 m), storing all intermediate results.
pub fn poly_zero_inner(
    unity: &[Fr],
    x: &[Fr],
    result: &mut [Vec<Fr>],
    n: usize,
    l: usize,
    r: usize,
) {
    if l + 1 == r {
        result[n] = vec![-x[l], Fr::ONE];
        return;
    }
    let m = l + r >> 1;
    poly_zero_inner(unity, x, result, n << 1, l, m);
    poly_zero_inner(unity, x, result, n << 1 | 1, m, r);
    let mut polyl = result[n << 1].clone();
    let mut polyr = result[n << 1 | 1].clone();
    poly_fft(unity, &mut polyl, r - l + 1);
    poly_fft(unity, &mut polyr, r - l + 1);
    result[n] = polyl.iter().zip(polyr.iter()).map(|(i, j)| i * j).collect();
    poly_ifft(unity, &mut result[n], r - l + 1);
    trim_zeroes(&mut result[n]);
}

/// Compute the zeroing polynomial in O(mlog^2 m).
pub fn poly_zero(unity: &[Fr], x: &[Fr]) -> Vec<Fr> {
    let logi = x.len().next_power_of_two().trailing_zeros();

    let mut poly: Vec<Vec<Fr>> = Vec::with_capacity(x.len());
    for i in x {
        poly.push(vec![-*i, Fr::ONE]);
    }
    for layer in 0..logi {
        let mut i = 0;
        while i < x.len() {
            if i + (1 << layer) < x.len() {
                // Check if the length of the polynomial overflows
                let overflow_flag = poly[i].len() == (1 << layer) + 1
                    && poly[i + (1 << layer)].len() == (1 << layer) + 1;

                poly_fft(unity, &mut poly[i], 1 << layer + 1);
                poly_fft(unity, &mut poly[i + (1 << layer)], 1 << layer + 1);
                poly[i] = poly[i]
                    .iter()
                    .zip(poly[i + (1 << layer)].iter())
                    .map(|(i, j)| i * j)
                    .collect();
                poly_ifft(unity, &mut poly[i], 1 << layer + 1);

                if overflow_flag {
                    poly[i][0] -= Fr::ONE;
                    poly[i].push(Fr::ONE);
                }
            }
            i += 1 << layer + 1;
        }
    }

    let mut p = poly.into_iter().next().expect("Vector is empty");
    trim_zeroes(&mut p);

    p
}

/// Compute delta, according to the paper
pub fn poly_delta(unity: &[Fr], x: &[Fr]) -> Vec<Fr> {
    let mut zero_poly = poly_zero(unity, x);
    for i in 0..zero_poly.len() - 1 {
        zero_poly[i] = zero_poly[i + 1] * Fr::from((i + 1) as u32);
    }
    zero_poly.pop();
    poly_evaluate(unity, &zero_poly, x)
}

/// Interpolate polynomial
pub fn poly_interpolate(unity: &[Fr], x: &[Fr], y: &[Fr]) -> Vec<Fr> {
    let logi = x.len().next_power_of_two().trailing_zeros();

    let d = poly_delta(unity, x);
    let mut f: Vec<Vec<Fr>> = Vec::with_capacity(x.len());
    let mut m: Vec<Vec<Fr>> = Vec::with_capacity(x.len());
    for i in 0..x.len() {
        f.push(vec![y[i] / d[i]]);
    }
    for i in x {
        m.push(vec![-*i, Fr::ONE]);
    }
    for layer in 0..logi {
        let mut i = 0;
        while i < x.len() {
            if i + (1 << layer) < x.len() {
                poly_fft(unity, &mut f[i], 1 << layer + 1);
                poly_fft(unity, &mut f[i + (1 << layer)], 1 << layer + 1);
                poly_fft(unity, &mut m[i], 1 << layer + 1);
                poly_fft(unity, &mut m[i + (1 << layer)], 1 << layer + 1);

                let mut fa = f[i]
                    .iter()
                    .zip(m[i + (1 << layer)].iter())
                    .map(|(i, j)| i * j)
                    .collect();
                let mut fb = f[i + (1 << layer)]
                    .iter()
                    .zip(m[i].iter())
                    .map(|(i, j)| i * j)
                    .collect();
                poly_ifft(unity, &mut fa, 1 << layer + 1);
                poly_ifft(unity, &mut fb, 1 << layer + 1);
                f[i].resize(max(fa.len(), fb.len()), Fr::ZERO);
                for j in 0..f[i].len() {
                    f[i][j] = (if j < fa.len() { fa[j] } else { Fr::ZERO })
                        + (if j < fb.len() { fb[j] } else { Fr::ZERO });
                }
                m[i] = m[i]
                    .iter()
                    .zip(m[i + (1 << layer)].iter())
                    .map(|(i, j)| i * j)
                    .collect();
                poly_ifft(unity, &mut m[i], 1 << layer + 1);
            }
            i += 1 << layer + 1;
        }
    }

    let mut p = f.into_iter().next().expect("Vector is empty");
    trim_zeroes(&mut p);
    p
}

/// Inverse some poly w.r.t. x^m
pub fn poly_inverse(unity: &[Fr], mut poly: Vec<Fr>, m: usize) -> Vec<Fr> {
    let n = m.next_power_of_two();
    poly.resize(n, Fr::ZERO);
    let mut res = vec![Fr::ZERO; n];
    res[0] = Fr::ONE / poly[0];
    poly_fft(unity, &mut poly, n);
    poly_fft(unity, &mut res, n);
    let mut degree = 1;
    let two = Fr::from(2);
    while degree < n {
        res = res
            .iter()
            .zip(poly.iter())
            .map(|(i, j)| (two - i * j) * i)
            .collect();
        degree <<= 1;
    }
    poly_ifft(unity, &mut res, n);
    res.truncate(m);
    trim_zeroes(&mut res);
    res
}

/// Poly division
pub fn poly_divide(unity: &[Fr], mut f: Vec<Fr>, mut g: Vec<Fr>) -> (Vec<Fr>, Vec<Fr>) {
    trim_zeroes(&mut f);
    trim_zeroes(&mut g);
    let n = f.len().next_power_of_two();
    let mut ff = f.clone();
    let mut gg = g.clone();
    ff.reverse();
    gg.reverse();
    let mut gi = poly_inverse(unity, gg, f.len() - g.len() + 1);
    poly_fft(unity, &mut ff, n);
    poly_fft(unity, &mut gi, n);
    let mut q = ff.iter().zip(gi.iter()).map(|(i, j)| i * j).collect();
    poly_ifft(unity, &mut q, n);
    q.resize(f.len() - g.len() + 1, Fr::ZERO);
    q.reverse();
    let qq = q.clone();

    poly_fft(unity, &mut q, n);
    poly_fft(unity, &mut g, n);
    let mut gq = g.iter().zip(q.iter()).map(|(i, j)| i * j).collect();
    poly_ifft(unity, &mut gq, n);
    let mut r: Vec<Fr> = Vec::with_capacity(f.len());
    for i in 0..f.len() {
        r.push(f[i] - (if i < gq.len() { gq[i] } else { Fr::ZERO }));
    }
    trim_zeroes(&mut r);
    (qq, r)
}

pub fn poly_evaluate_inner(
    unity: &[Fr],
    poly: &[Fr],
    x: &[Fr],
    zero: &[Vec<Fr>],
    result: &mut [Fr],
    n: usize,
    l: usize,
    r: usize,
) {
    if l + 1 == r {
        result[l] = poly
            .iter()
            .rev()
            .fold(Fr::default(), |acc, &coeff| acc * x[l] + coeff);
        return;
    }
    let m = l + r >> 1;
    let (_ql, rl) = poly_divide(unity, poly.to_vec(), zero[n << 1].clone());
    let (_qr, rr) = poly_divide(unity, poly.to_vec(), zero[n << 1 | 1].clone());
    poly_evaluate_inner(unity, &rl, x, zero, result, n << 1, l, m);
    poly_evaluate_inner(unity, &rr, x, zero, result, n << 1 | 1, m, r);
}

/// Evaluate poly at a list of points
pub fn poly_evaluate(unity: &[Fr], poly: &[Fr], x: &[Fr]) -> Vec<Fr> {
    let mut zero = vec![Vec::<Fr>::new(); x.len() << 2];
    poly_zero_inner(unity, x, &mut zero, 1, 0, x.len());
    let mut result = vec![Fr::ZERO; x.len()];
    poly_evaluate_inner(unity, poly, x, &zero, &mut result, 1, 0, x.len());
    result
}

#[cfg(test)]
mod tests {
    use ark_bls12_381::G1Projective as G1;
    use ark_ec::Group;

    use super::*;
    #[test]
    fn test_poly_evaluate() {
        let unity = compute_unity(1 << 3);
        assert_eq!(
            poly_evaluate(
                &unity,
                &[3, 2, 1].map(|i| Fr::from(i)),
                &[1, 2, 3].map(|i| Fr::from(i))
            ),
            [6, 11, 18].map(|i| Fr::from(i)).to_vec()
        );
    }

    #[test]
    fn test_poly_zero_inner() {
        let unity = compute_unity(1 << 3);
        let mut result = vec![Vec::<Fr>::new(); 8 << 2];
        let x = [1, 2, 4, 5, 6].map(|i| Fr::from(i));
        poly_zero_inner(&unity, &x, &mut result, 1, 0, 5);
        let z = poly_zero(&unity, &x);
        assert_eq!(z, result[1]);
    }

    #[test]
    fn test_poly() {
        let unity = compute_unity(1 << 3);
        assert_eq!(
            poly_inverse(&unity, [1, 1].map(Fr::from).to_vec(), 2),
            [1, -1].map(Fr::from).to_vec()
        );
        assert_eq!(
            poly_divide(
                &unity,
                [3, 5, 2].map(Fr::from).to_vec(),
                [2, 1].map(Fr::from).to_vec(),
            ),
            ([1, 2].map(Fr::from).to_vec(), [1].map(Fr::from).to_vec(),)
        );

        assert_eq!(
            poly_zero(&unity, &[1, 2].map(|i| unity[i])),
            [unity[3], -unity[1] - unity[2], Fr::ONE].to_vec()
        );
        assert_eq!(
            poly_delta(&unity, &[1, 2, 3].map(|i| unity[i])),
            [
                (unity[1] - unity[2]) * (unity[1] - unity[3]),
                (unity[2] - unity[3]) * (unity[2] - unity[1]),
                (unity[3] - unity[1]) * (unity[3] - unity[2])
            ]
            .to_vec()
        );
        assert_eq!(
            poly_interpolate(
                &unity,
                &[1, 2].map(|i| unity[i]),
                &[unity[2], unity[1]].map(Fr::from)
            ),
            [unity[1] + unity[2], -Fr::ONE].to_vec()
        );

        let mut v: Vec<G1> = [2, 4, 3, 1].map(|i| G1::generator() * Fr::from(i)).to_vec();
        let vv = v.clone();
        poly_fft(&unity, &mut v, 4);
        poly_ifft(&unity, &mut v, 4);
        assert_eq!(v, vv);
    }
}
