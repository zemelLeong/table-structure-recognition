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
            .with_key_remap("model.ax.2", "model.ax.1")
            .with_key_remap("model.ax.4", "model.ax.2")
            .with_key_remap("model.ax.6", "model.ax.3")
            .with_key_remap("model.ax.8", "model.ax.4")
            .with_key_remap("model.ax", "model.ax.layers")

            .with_key_remap("model.cr.2", "model.cr.1")
            .with_key_remap("model.cr.4", "model.cr.2")
            .with_key_remap("model.cr.6", "model.cr.3")
            .with_key_remap("model.cr.8", "model.cr.4")
            .with_key_remap("model.cr", "model.cr.layers")

            .with_key_remap("model.hm.2", "model.hm.1")
            .with_key_remap("model.hm.4", "model.hm.2")
            .with_key_remap("model.hm.6", "model.hm.3")
            .with_key_remap("model.hm.8", "model.hm.4")
            .with_key_remap("model.hm", "model.hm.layers")

            .with_key_remap("model.reg.2", "model.reg.1")
            .with_key_remap("model.reg.4", "model.reg.2")
            .with_key_remap("model.reg.6", "model.reg.3")
            .with_key_remap("model.reg.8", "model.reg.4")
            .with_key_remap("model.reg", "model.reg.layers")

            .with_key_remap("model.st.2", "model.st.1")
            .with_key_remap("model.st.4", "model.st.2")
            .with_key_remap("model.st.6", "model.st.3")
            .with_key_remap("model.st.8", "model.st.4")
            .with_key_remap("model.st", "model.st.layers")

            .with_key_remap("model.wh.2", "model.wh.1")
            .with_key_remap("model.wh.4", "model.wh.2")
            .with_key_remap("model.wh.6", "model.wh.3")
            .with_key_remap("model.wh.8", "model.wh.4")
            .with_key_remap("model.wh", "model.wh.layers")
            .with_key_remap("model.adaptionU1", "model.adaption_u1")
            
            .with_key_remap(
                "processor.stacker.tsfm.decoder.linear.2",
                "processor.stacker.tsfm.decoder.linear.1",
            )
            .with_key_remap(
                "processor.stacker.tsfm.decoder.linear",
                "processor.stacker.tsfm.decoder.linear.layers",
            )
            
            .with_key_remap(
                "processor.tsfm_axis.decoder.linear.2",
                "processor.tsfm_axis.decoder.linear.1",
            )
            .with_key_remap(
                "processor.tsfm_axis.decoder.linear",
                "processor.tsfm_axis.decoder.linear.layers",
            )
            
            .with_key_remap(
                "processor.stacker.logi_encoder.2",
                "processor.stacker.logi_encoder.1",
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
