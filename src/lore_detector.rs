use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig};
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, PaddingConfig2d};
use burn::tensor::activation::relu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use std::collections::HashMap;

const BN_MOMENTUM: f64 = 0.1;

#[derive(Module, Debug)]
struct Sequential<B: Backend> {
    layers: Vec<BasicBlock<B>>,
}

impl<B: Backend> Sequential<B> {
    fn new() -> Self {
        Self { layers: Vec::new() }
    }

    fn add(&mut self, layer: BasicBlock<B>) {
        self.layers.push(layer);
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(x);
        }
        x
    }
}

#[derive(Module, Debug)]
struct SequentialDeconv<B: Backend> {
    layers: Vec<DeconvLayer<B>>,
}

impl<B: Backend> SequentialDeconv<B> {
    fn new() -> Self {
        Self { layers: Vec::new() }
    }

    fn add(&mut self, layer: DeconvLayer<B>) {
        self.layers.push(layer);
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(x);
        }
        x
    }
}

#[derive(Module, Debug)]
struct SequentialConv2d<B: Backend> {
    layers: Vec<Conv2d<B>>,
}

impl<B: Backend> SequentialConv2d<B> {
    fn new() -> Self {
        Self { layers: Vec::new() }
    }

    fn from(conv2ds: Vec<Conv2d<B>>) -> Self {
        Self { layers: conv2ds }
    }

    fn add(&mut self, layer: Conv2d<B>) {
        self.layers.push(layer);
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut x = x;
        for (index, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x);
            if index < self.layers.len() - 1 {
                x = relu(x);
            }
        }
        x
    }
}

#[derive(Module, Debug)]
pub enum Layers<B: Backend> {
    Conv2d(Conv2d<B>),
    BatchNorm(BatchNorm<B, 2>),
    ConvTranspose2d(ConvTranspose2d<B>),
}

impl<B: Backend> Layers<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        match self {
            Layers::Conv2d(c) => c.forward(x),
            Layers::BatchNorm(b) => b.forward(x),
            Layers::ConvTranspose2d(ct) => ct.forward(x),
        }
    }
}

#[derive(Module, Debug)]
pub struct DownSample<B: Backend> {
    layers: Vec<Layers<B>>,
}

#[derive(Module, Debug)]
pub struct DeconvLayer<B: Backend> {
    layers: Vec<Layers<B>>,
}

impl<B: Backend> DeconvLayer<B> {
    pub fn new_with(
        inplanes: usize,
        planes: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        output_padding: usize,
        record: DeconvLayerRecord<B>,
    ) -> Self {
        let layers = record
            .layers
            .into_iter()
            .map(|l| match l {
                LayersRecord::BatchNorm(b) => {
                    let bn = BatchNormConfig::new(planes)
                        .with_momentum(BN_MOMENTUM)
                        .init_with(b);
                    Layers::BatchNorm(bn)
                }
                LayersRecord::ConvTranspose2d(c) => {
                    let ct = ConvTranspose2dConfig::new([inplanes, planes], [kernel, kernel])
                        .with_stride([stride, stride])
                        .with_padding([padding, padding])
                        .with_padding_out([output_padding, output_padding])
                        .with_bias(false)
                        .init_with(c);
                    Layers::ConvTranspose2d(ct)
                }
                _ => panic!("Invalid layer"),
            })
            .collect();
        Self { layers }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut out = x;
        for layer in self.layers.iter() {
            match layer {
                Layers::ConvTranspose2d(ct) => out = ct.forward(out),
                Layers::BatchNorm(bn) => out = bn.forward(out),
                _ => panic!("Invalid layer"),
            }
        }
        relu(out)
    }
}

impl<B: Backend> DownSample<B> {
    pub fn new_with(
        inplanes: usize,
        planes: usize,
        stride: usize,
        record: DownSampleRecord<B>,
    ) -> Self {
        let layers = record
            .layers
            .into_iter()
            .map(|l| match l {
                LayersRecord::Conv2d(c) => {
                    let conv = Conv2dConfig::new([inplanes, planes], [1, 1])
                        .with_stride([stride, stride])
                        .with_bias(false)
                        .init_with(c);
                    Layers::Conv2d(conv)
                }
                LayersRecord::BatchNorm(b) => {
                    let bn = BatchNormConfig::new(planes)
                        .with_momentum(BN_MOMENTUM)
                        .init_with(b);
                    Layers::BatchNorm(bn)
                }
                _ => panic!("Invalid layer"),
            })
            .collect();
        Self { layers }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut out = x;
        for layer in self.layers.iter() {
            out = layer.forward(out);
        }
        out
    }
}

#[derive(Module, Debug)]
pub struct BasicBlock<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B, 2>,
    downsample: Option<DownSample<B>>,
    stride: usize,
    planes: usize,
}

impl<B: Backend> BasicBlock<B> {
    pub fn new_with(
        inplanes: usize,
        planes: usize,
        stride: usize,
        need_downsample: bool,
        record: BasicBlockRecord<B>,
    ) -> Self {
        let conv1 = Conv2dConfig::new([inplanes, planes], [3, 3])
            .with_stride([stride, stride])
            .init_with(record.conv1);
        let bn1 = BatchNormConfig::new(planes)
            .with_momentum(BN_MOMENTUM)
            .init_with(record.bn1);
        let conv2 = Conv2dConfig::new([planes, planes], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init_with(record.conv2);
        let bn2 = BatchNormConfig::new(planes)
            .with_momentum(BN_MOMENTUM)
            .init_with(record.bn2);

        let downsample = if need_downsample {
            let ds_record = record.downsample.expect("No downsample layer in record");
            let ds = DownSample::new_with(inplanes, planes, stride, ds_record);
            Some(ds)
        } else {
            None
        };

        Self {
            conv1,
            bn1,
            conv2,
            bn2,
            downsample,
            stride,
            planes,
        }
    }

    /// ToDo Tensor 维度可能有问题，待测试时再修改
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut residual = x.clone();
        let out = self.conv1.forward(x);
        let out = self.bn1.forward(out);
        let out = relu(out);
        let out = self.conv2.forward(out);
        let out = self.bn2.forward(out);
        if let Some(downsample) = &self.downsample {
            residual = downsample.forward(residual);
        }
        relu(out + residual)
    }
}

#[derive(Module, Debug)]
pub struct LoreDetectModel<B: Backend> {
    inplanes: usize,
    deconv_with_bias: bool,
    layers: [usize; 4],
    head_conv: usize,
    conv1: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    maxpool: MaxPool2d,
    layer1: Sequential<B>,
    layer2: Sequential<B>,
    layer3: Sequential<B>,
    layer4: Sequential<B>,
    adaption3: Conv2d<B>,
    adaption2: Conv2d<B>,
    adaption1: Conv2d<B>,
    adaption0: Conv2d<B>,
    adaption_u1: Conv2d<B>,
    deconv_layers1: SequentialDeconv<B>,
    deconv_layers2: SequentialDeconv<B>,
    deconv_layers3: SequentialDeconv<B>,
    deconv_layers4: SequentialDeconv<B>,
    hm_maxpool: MaxPool2d,
    mk_maxpool: MaxPool2d,
    ax: SequentialConv2d<B>,
    cr: SequentialConv2d<B>,
    hm: SequentialConv2d<B>,
    reg: SequentialConv2d<B>,
    st: SequentialConv2d<B>,
    wh: SequentialConv2d<B>,
}

#[derive(Module, Debug)]
pub struct LoreDetectModelS<B: Backend> {
    layer1: Sequential<B>,
}

impl<B: Backend> LoreDetectModel<B> {
    fn get_heads() -> HashMap<String, usize> {
        HashMap::from([
            ("ax".to_owned(), 256),
            ("cr".to_owned(), 256),
            ("hm".to_owned(), 2),
            ("reg".to_owned(), 2),
            ("st".to_owned(), 8),
            ("wh".to_owned(), 8),
        ])
    }

    pub fn new_with(record: LoreDetectModelRecord<B>) -> Self {
        let inplanes = 64;
        let deconv_with_bias = false;
        let layers = [2, 2, 2, 2];
        let head_conv = 64;
        let conv1 = Conv2dConfig::new([3, 64], [7, 7])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(3, 3))
            .init_with(record.conv1);
        let bn1 = BatchNormConfig::new(64)
            .with_momentum(BN_MOMENTUM)
            .init_with(record.bn1);
        let maxpool = MaxPool2dConfig::new([3, 3])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init();

        let adaption3 = Conv2dConfig::new([256, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 0))
            .with_bias(false)
            .init_with(record.adaption3);
        let adaption2 = Conv2dConfig::new([128, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 0))
            .with_bias(false)
            .init_with(record.adaption2);
        let adaption1 = Conv2dConfig::new([64, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 0))
            .with_bias(false)
            .init_with(record.adaption1);
        let adaption0 = adaption1.clone();
        let adaption_u1 = adaption1.clone();
        let hm_maxpool = MaxPool2dConfig::new([3, 3])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init();
        let mk_maxpool = hm_maxpool.clone();

        let layer1 = Sequential::new();
        let layer2 = Sequential::new();
        let layer3 = Sequential::new();
        let layer4 = Sequential::new();

        let deconv_layers1 = SequentialDeconv::new();
        let deconv_layers2 = SequentialDeconv::new();
        let deconv_layers3 = SequentialDeconv::new();
        let deconv_layers4 = SequentialDeconv::new();

        let input_channels = 256;
        let mut ax = SequentialConv2d::new();
        let mut cr = SequentialConv2d::new();
        let mut hm = SequentialConv2d::new();
        let mut reg = SequentialConv2d::new();
        let mut st = SequentialConv2d::new();
        let mut wh = SequentialConv2d::new();

        let max_len = 5;
        for (index, r) in record.ax.layers.into_iter().enumerate() {
            let ks = if index == max_len - 1 { 1 } else { 3 };
            let conv = Conv2dConfig::new([input_channels, head_conv], [ks, ks])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init_with(r);
            ax.add(conv);
        }

        for (index, r) in record.cr.layers.into_iter().enumerate() {
            let ks = if index == max_len - 1 { 1 } else { 3 };
            let conv = Conv2dConfig::new([input_channels, head_conv], [ks, ks])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init_with(r);
            cr.add(conv);
        }

        for (index, r) in record.hm.layers.into_iter().enumerate() {
            let ks = if index == max_len - 1 { 1 } else { 3 };
            let conv = Conv2dConfig::new([input_channels, head_conv], [ks, ks])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init_with(r);
            hm.add(conv);
        }

        for (index, r) in record.st.layers.into_iter().enumerate() {
            let ks = if index == max_len - 1 { 1 } else { 3 };
            let conv = Conv2dConfig::new([input_channels, head_conv], [ks, ks])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init_with(r);
            st.add(conv);
        }

        for (index, r) in record.wh.layers.into_iter().enumerate() {
            let ks = if index == max_len - 1 { 1 } else { 3 };
            let conv = Conv2dConfig::new([input_channels, head_conv], [ks, ks])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init_with(r);
            wh.add(conv);
        }

        for r in record.reg.layers.into_iter() {
            let conv = Conv2dConfig::new([input_channels, head_conv], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init_with(r);
            reg.add(conv);
        }

        let mut lore_dector = Self {
            inplanes,
            deconv_with_bias,
            layers,
            head_conv,
            conv1,
            bn1,
            maxpool,
            layer1,
            layer2,
            layer3,
            layer4,
            adaption3,
            adaption2,
            adaption1,
            adaption0,
            adaption_u1,
            deconv_layers1,
            deconv_layers2,
            deconv_layers3,
            deconv_layers4,
            hm_maxpool,
            mk_maxpool,
            ax,
            cr,
            hm,
            reg,
            st,
            wh,
        };

        let layer1 = lore_dector.make_layer(64, 2, Some(2), record.layer1);
        let layer2 = lore_dector.make_layer(128, 2, Some(2), record.layer2);
        let layer3 = lore_dector.make_layer(256, 2, Some(2), record.layer3);
        let layer4 = lore_dector.make_layer(256, 2, Some(2), record.layer4);
        let deconv_layers1 =
            lore_dector.make_deconv_layer(1, vec![256], vec![4], record.deconv_layers1);
        let deconv_layers2 =
            lore_dector.make_deconv_layer(1, vec![256], vec![4], record.deconv_layers2);
        let deconv_layers3 =
            lore_dector.make_deconv_layer(1, vec![256], vec![4], record.deconv_layers3);
        let deconv_layers4 =
            lore_dector.make_deconv_layer(1, vec![256], vec![4], record.deconv_layers4);

        Self {
            layer1,
            layer2,
            layer3,
            layer4,
            deconv_layers1,
            deconv_layers2,
            deconv_layers3,
            deconv_layers4,
            ..lore_dector
        }
    }

    fn make_layer(
        &mut self,
        planes: usize,
        blocks: usize,
        stride: Option<usize>,
        record: SequentialRecord<B>,
    ) -> Sequential<B> {
        let mut record = record;
        let first_layer = record.layers.remove(0);

        let stride = stride.unwrap_or(1);

        let mut sequential = Sequential::new();
        let need_downsample = stride != 1 || self.inplanes != planes;

        sequential.add(BasicBlock::new_with(
            self.inplanes,
            planes,
            stride,
            need_downsample,
            first_layer,
        ));

        self.inplanes = planes;
        assert_eq!(blocks - 1, record.layers.len());
        for _ in 1..blocks {
            let block_record = record.layers.remove(0);
            let block = BasicBlock::new_with(self.inplanes, planes, stride, false, block_record);
            sequential.add(block);
        }
        sequential
    }

    fn make_deconv_layer(
        &mut self,
        num_layers: usize,
        num_filters: Vec<usize>,
        num_kernels: Vec<usize>,
        record: SequentialDeconvRecord<B>,
    ) -> SequentialDeconv<B> {
        let msg = "ERROR: num_deconv_layers is different len(num_deconv_filters)";
        assert_eq!(num_layers, num_filters.len(), "{}", msg);
        assert_eq!(num_layers, num_kernels.len(), "{}", msg);

        let mut layers = SequentialDeconv::new();
        for r in record.layers.into_iter() {
            let kernel = num_kernels[0];
            let padding = match kernel {
                4 => 1,
                3 => 1,
                2 => 0,
                7 => 3,
                _ => panic!("Invalid kernel size"),
            };
            let output_padding = match kernel {
                4 => 0,
                3 => 1,
                2 => 0,
                7 => 0,
                _ => panic!("Invalid kernel size"),
            };
            let planes = num_filters[0];
            let deconv =
                DeconvLayer::new_with(self.inplanes, planes, kernel, 2, padding, output_padding, r);
            layers.add(deconv);
            self.inplanes = planes;
        }
        layers
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> HashMap<String, Tensor<B, 4>> {
        let x = self.conv1.forward(x);
        let x = self.bn1.forward(x);
        let x = relu(x);
        let x0 = self.maxpool.forward(x);
        let x1 = self.layer1.forward(x0.clone());
        let x2 = self.layer2.forward(x1.clone());
        let x3 = self.layer3.forward(x2.clone());
        let x4 = self.layer4.forward(x3.clone());

        let x3_ = self.deconv_layers1.forward(x4);
        let x3_ = self.adaption3.forward(x3) + x3_;

        let x2_ = self.deconv_layers2.forward(x3_);
        let x2_ = self.adaption2.forward(x2) + x2_;

        let x1_ = self.deconv_layers3.forward(x2_);
        let x1_ = self.adaption1.forward(x1) + x1_;

        let x0_ = self.deconv_layers4.forward(x1_) + self.adaption0.forward(x0);
        let x0_ = self.adaption_u1.forward(x0_);

        let mut ret = HashMap::new();
        ret.insert("ax".to_owned(), self.ax.forward(x0_.clone()));
        ret.insert("cr".to_owned(), self.cr.forward(x0_.clone()));
        ret.insert("hm".to_owned(), self.hm.forward(x0_.clone()));
        ret.insert("reg".to_owned(), self.reg.forward(x0_.clone()));
        ret.insert("st".to_owned(), self.st.forward(x0_.clone()));
        ret.insert("wh".to_owned(), self.wh.forward(x0_));

        ret
    }
}
