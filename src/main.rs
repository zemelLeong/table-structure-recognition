use crate::model_lore::LoreModel;

mod lore_detector;
mod lore_processor;
mod model_lore;

type Backend = burn_ndarray::NdArray<f64>;

fn main() {
    let model_path = "./files/cv_resnet-transformer_table-structure-recognition_lore.pt";
    let device = Default::default();
    LoreModel::<Backend>::new(model_path, &device);
}
