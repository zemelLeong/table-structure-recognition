use crate::lore_detector::LoreDetectModel;
use crate::lore_processor::LoreProcessModel;
use burn::module::Module;
use burn::record::{FullPrecisionSettings, Recorder};
use burn::tensor::backend::Backend;
use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};

#[derive(Module, Debug)]
pub struct LoreModel<B: Backend> {
    model: LoreDetectModel<B>,
    processor: LoreProcessModel<B>,
}

impl<B: Backend> LoreModel<B> {
    pub fn new(model_path: &str, device: &B::Device) -> Self {
        let load_args = LoadArgs::new(model_path.into())
            .with_key_remap("model.wh", "model.wh.layers")
            .with_key_remap("model.ax", "model.ax.layers")
            .with_key_remap("model.cr", "model.cr.layers")
            .with_key_remap("model.hm", "model.hm.layers")
            .with_key_remap("model.reg", "model.reg.layers")
            .with_key_remap("model.st", "model.st.layers")
            .with_key_remap(
                "processor.stacker.tsfm.decoder.linear",
                "processor.stacker.tsfm.decoder.linear.layers",
            )
            .with_key_remap(
                "processor.tsfm_axis.decoder.linear",
                "processor.tsfm_axis.decoder.linear.layers",
            )
            .with_key_remap(
                "processor.stacker.logi_encoder",
                "processor.stacker.logi_encoder.layers",
            )
            .with_key_remap(r"(model.layer\d)", r"$1.layers")
            .with_key_remap(r"(model.deconv_layers\d)", r"$1.layers")
            .with_key_remap("downsample.0", "downsample.conv")
            .with_key_remap("downsample.1", "downsample.bn");
        let record: LoreModelRecord<B> = PyTorchFileRecorder::<FullPrecisionSettings>::default()
            .load(load_args, device)
            .unwrap();

        let model = LoreDetectModel::new_with(record.model);
        let processor = LoreProcessModel::new_with(record.processor, device);

        Self { model, processor }
    }
}
