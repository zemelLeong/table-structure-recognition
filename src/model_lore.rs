use crate::lore_detector::LoreDetectModel;
use crate::lore_processor::LoreProcessModel;
use burn::module::Module;
use burn::record::{FullPrecisionSettings, Recorder};
use burn::tensor::backend::Backend;
use burn_import::pytorch::PyTorchFileRecorder;

#[derive(Module, Debug)]
pub struct LoreModel<B: Backend> {
    model: LoreDetectModel<B>,
    processor: LoreProcessModel<B>,
}

impl<B: Backend> LoreModel<B> {
    pub fn new(model_path: &str, device: &B::Device) -> Self {
        let record: LoreModelRecord<B> = PyTorchFileRecorder::<FullPrecisionSettings>::default()
            .load(model_path.into(), device)
            .unwrap();

        let model = LoreDetectModel::new_with(record.model);
        let processor = LoreProcessModel::new_with(record.processor, device);

        Self { model, processor }
    }
}
