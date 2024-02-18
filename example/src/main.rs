use galois::{DTensor, Shape, Tensor};
use lightnn::{Model, Tensor as lnTensor};
use smallvec::smallvec;
fn main() {
    let image = image::open("/opt/rsproject/gptgrep/lightnn/example/grace_hopper.jpg")
        .unwrap()
        .to_rgb8();
    let resized =
        image::imageops::resize(&image, 224, 224, ::image::imageops::FilterType::Triangle);
    let m1: DTensor<f32> = DTensor::with_shape_fn(Shape::from_array([1, 3, 224, 224]), |s| {
        let (i, c, y, x) = s.dims4();
        let mean = [0.485, 0.456, 0.406][c];
        let std = [0.229, 0.224, 0.225][c];
        (resized[(x as _, y as _)][c] as f32 / 255.0 - mean) / std
    });

    let model =
        Model::from_path("/opt/rsproject/gptgrep/lightnn/model/mobilenetv2-7.onnx").unwrap();
    let mut session = model.session().unwrap();
    let t = lnTensor::new(Tensor::F32(m1));
    session.set_input(smallvec![t]);
    let t = session.run().unwrap();
    println!("{:?}", t[0].as_value_ref().as_tensor_ref());
}
