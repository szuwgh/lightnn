use ndarray::Array;
use ndarray::ArrayD;
use ndarray::IxDyn;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use rand_isaac::isaac64::Isaac64Rng;

struct Network {
    num_layers: usize,
    sizes: Vec<u32>,
    weights: ArrayD<_>,
}

impl Network {
    fn new(sizes: &[u32]) -> Network {
        let a = sizes
            .iter()
            .zip(sizes[1..].iter())
            .map(|(x, y)| Array::random((3, 2), Normal::<f64>::new(0., 1.).unwrap()))
            .collect::<Vec<_>>();
        Self {
            num_layers: sizes.len(),
            sizes: sizes.to_vec(),
            weights: ArrayD::from_vec(a),
        }
    }

    fn feedforward() {
        // Return the output of the network if ``a`` is input.
    }

    fn SGD() {}
}

fn sigmoid(z: f64) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network() {
        //let xs = [2, 3, 1];
        let a = Array::random((3, 2), Normal::<f64>::new(0., 1.).unwrap());
        println!("{}", a);
    }
}
