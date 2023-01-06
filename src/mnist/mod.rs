use std::ops::Div;

use image::*;
use mnist::*;
use ndarray::prelude::*;

fn load_mnist() {
    let (trn_size, _rows, _cols) = (50_000, 28, 28);
    // Deconstruct the returned Mnist struct.
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .base_path("/opt/rsproject/lightnn/src/data/")
        .label_format_digit()
        .training_set_length(trn_size)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    let image_num = 0;
    // Can use an Array2 or Array3 here (Array3 for visualization)
    let train_data = Array3::from_shape_vec((50_000, 28, 28), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.0);
    println!("{:#.1?}\n", train_data.slice(s![image_num, .., ..]));

    // Convert the returned Mnist struct to Array2 format
    let train_labels: Array2<f32> = Array2::from_shape_vec((50_000, 1), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f32);
    println!(
        "The first digit is a {:?}",
        train_labels.slice(s![image_num, ..])
    );

    let _test_data = Array3::from_shape_vec((10_000, 28, 28), tst_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.);

    let _test_labels: Array2<f32> = Array2::from_shape_vec((10_000, 1), tst_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as f32);

    let item_num = 49999;
    let x = train_data.slice(s![item_num, .., ..]).to_owned();
    //    / x.div(rhs)
    println!("x len {:?}", x);

    let image = bw_ndarray2_to_rgb_image(train_data.slice(s![item_num, .., ..]).to_owned());

    image
        .save("/opt/rsproject/lightnn/src/data/testimage.png")
        .unwrap();
}

fn bw_ndarray2_to_rgb_image(arr: Array2<f32>) -> RgbImage {
    assert!(arr.is_standard_layout());

    let (width, height) = (arr.ncols(), arr.ncols());
    let mut img: RgbImage = ImageBuffer::new(width as u32, height as u32);
    for y in 0..height {
        for x in 0..width {
            let val = (arr[[y, x]] * 255.) as u8;
            img.put_pixel(x as u32, y as u32, image::Rgb([val, val, val]))
        }
    }
    img
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mnists() {
        load_mnist()
    }
}
