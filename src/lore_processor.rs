use burn::module::{Module, Param, ParamId};
use burn::nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::activation::{relu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{Data, Int, Shape, Tensor};

#[derive(Module, Debug)]
struct SequentialLinear<B: Backend> {
    layers: Vec<Linear<B>>,
}

impl<B: Backend> SequentialLinear<B> {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn add(&mut self, layer: Linear<B>) {
        self.layers.push(layer);
    }

    pub fn from(layers: Vec<Linear<B>>) -> Self {
        Self { layers }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut output = input;
        for (index, layer) in self.layers.iter().enumerate() {
            output = layer.forward(output);
            if index < self.layers.len() - 1 {
                output = relu(output);
            }
        }
        output
    }
}

#[derive(Module, Debug)]
struct Stacker<B: Backend> {
    logi_encoder: SequentialLinear<B>,
    tsfm: Transformer<B>,
}

impl<B: Backend> Stacker<B> {
    pub fn new_with(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        layers: usize,
        heads: Option<usize>,
        dropout: Option<f64>,
        device: &B::Device,
        record: StackerRecord<B>,
    ) -> Self {
        let mut le = record.logi_encoder;
        let heads = heads.unwrap_or(8);
        let logic_encoder = SequentialLinear::from(vec![
            LinearConfig::new(input_size, hidden_size).init_with(le.layers.remove(0)),
            LinearConfig::new(hidden_size, hidden_size).init_with(le.layers.remove(0)),
        ]);

        Self {
            logi_encoder: logic_encoder,
            tsfm: Transformer::new_with(
                2 * hidden_size,
                hidden_size,
                output_size,
                layers,
                heads,
                dropout,
                device,
                record.tsfm,
            ),
        }
    }
    pub fn forward(&self, outputs: Tensor<B, 3>, logi: Tensor<B, 3>) -> Tensor<B, 3> {
        let logi_embeddings = relu(self.logi_encoder.forward(logi));
        let cat_embeddings = Tensor::cat(vec![logi_embeddings, outputs], 2);

        self.tsfm.forward(cat_embeddings)
    }
}

#[derive(Module, Debug)]
struct PositionalEncoder<B: Backend> {
    d_model: usize,
    dropout: Dropout,
    pe: Tensor<B, 3>,
}

impl<B: Backend> PositionalEncoder<B> {
    pub fn new_with(d_model: usize, max_seq_len: Option<usize>, dropout: Option<f64>) -> Self {
        let max_seq_len = max_seq_len.unwrap_or(900);
        let pe = vec![vec![0.0; d_model]; max_seq_len];

        let pe = pe
            .iter()
            .enumerate()
            .flat_map(|(pos, arr)| {
                arr.iter().enumerate().map(move |(i, _)| {
                    if i % 2 == 0 {
                        let sin_coef = (10000.0_f64).powf((2 * i) as f64 / d_model as f64);
                        (pos as f64 / sin_coef).sin()
                    } else {
                        let cos_coef = (10000.0_f64).powf((2 * (i + 1)) as f64 / d_model as f64);
                        (pos as f64 / cos_coef).cos()
                    }
                })
            })
            .collect::<Vec<f64>>();
        let pe = Tensor::<B, 2>::from_data(
            Data::new(pe, Shape::new([max_seq_len, d_model])).convert(),
            &B::Device::default(),
        );
        let pe = pe.unsqueeze::<3>();
        let dropout = DropoutConfig::new(dropout.unwrap_or(0.1)).init();
        Self {
            d_model,
            dropout,
            pe,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = x * (self.d_model as f64).sqrt();
        let pe = self
            .pe
            .clone()
            .set_require_grad(false)
            .to_device(&x.device());
        let x = x + pe;
        self.dropout.forward(x)
    }
}

#[derive(Module, Debug)]
struct Norm<B: Backend> {
    size: usize,
    alpha: Param<Tensor<B, 1>>,
    bias: Param<Tensor<B, 1>>,
    eps: f64,
}

impl<B: Backend> Norm<B> {
    pub fn new_with(d_model: usize, eps: Option<f64>, device: &B::Device) -> Self {
        let eps = eps.unwrap_or(1e-6);
        let alpha = Param::new(ParamId::from("alpha"), Tensor::ones([d_model], device));
        let bias = Param::new(ParamId::from("bias"), Tensor::zeros([d_model], device));
        Self {
            size: d_model,
            alpha,
            bias,
            eps,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mean = x.clone().mean_dim(2);
        let std = x.clone().var(2).sqrt();

        (x - mean) / (std + self.eps) * self.alpha.val().into_scalar()
            + self.bias.val().into_scalar()
    }
}

#[derive(Module, Debug)]
struct MultiHeadAttention<B: Backend> {
    d_model: usize,
    d_k: usize,
    h: usize,
    q_linear: Linear<B>,
    v_linear: Linear<B>,
    k_linear: Linear<B>,
    dropout: Dropout,
    out: Linear<B>,
}

impl<B: Backend> MultiHeadAttention<B> {
    pub fn new_with(
        d_model: usize,
        heads: usize,
        dropout: Option<f64>,
        record: MultiHeadAttentionRecord<B>,
    ) -> Self {
        let d_k = d_model / heads;

        let q_linear = LinearConfig::new(d_model, d_model).init_with(record.q_linear);
        let v_linear = LinearConfig::new(d_model, d_model).init_with(record.v_linear);
        let k_linear = LinearConfig::new(d_model, d_model).init_with(record.k_linear);
        let out = LinearConfig::new(d_model, d_model).init_with(record.out);

        let dropout = DropoutConfig::new(dropout.unwrap_or(0.1)).init();
        Self {
            d_model,
            d_k,
            h: heads,
            q_linear,
            v_linear,
            k_linear,
            dropout,
            out,
        }
    }

    pub fn attention_map(&self, q: Tensor<B, 3>, k: Tensor<B, 3>) -> Tensor<B, 4> {
        let bs = q.dims()[0];
        let sl = q.dims()[1];

        let k = self.k_linear.forward(k).reshape([bs, sl, self.h, self.d_k]);
        let q = self.q_linear.forward(q).reshape([bs, sl, self.h, self.d_k]);

        let k = k.swap_dims(1, 2);
        let q = q.swap_dims(1, 2);

        self.attention_score(q, k)
    }

    pub fn attention_score(&self, q: Tensor<B, 4>, k: Tensor<B, 4>) -> Tensor<B, 4> {
        let scores = q.matmul(k.swap_dims(2, 3)) / (self.d_k as f64).sqrt();
        softmax(scores, 2)
    }

    pub fn attention(&self, q: Tensor<B, 3>, k: Tensor<B, 3>, v: Tensor<B, 3>) -> Tensor<B, 3> {
        let scores = q.matmul(k.swap_dims(1, 2)) / (self.d_k as f64).sqrt();
        let scores = softmax(scores, 2);
        let scores = self.dropout.forward(scores);

        scores.matmul(v)
    }

    pub fn forward(&self, q: Tensor<B, 3>, k: Tensor<B, 3>, v: Tensor<B, 3>) -> Tensor<B, 3> {
        let bs = q.dims()[0];
        let sl = q.dims()[1];

        let k = self.k_linear.forward(k).reshape([bs, sl, self.h, self.d_k]);
        let q = self.q_linear.forward(q).reshape([bs, sl, self.h, self.d_k]);
        let v = self.v_linear.forward(v).reshape([bs, sl, self.h, self.d_k]);

        let k = k.swap_dims(1, 2);
        let q = q.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        let scores = self.attention_score(q, k);
        let scores = self.dropout.forward(scores);
        let output = scores.matmul(v);
        let output = output.swap_dims(1, 2).reshape([bs, sl, self.d_model]);
        self.out.forward(output)
    }
}

#[derive(Module, Debug)]
struct FeedForward<B: Backend> {
    linear_1: Linear<B>,
    linear_2: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> FeedForward<B> {
    pub fn new_with(
        d_model: usize,
        d_ff: Option<usize>,
        dropout: Option<f64>,
        record: FeedForwardRecord<B>,
    ) -> Self {
        let d_ff = d_ff.unwrap_or(2048);
        let linear1 = LinearConfig::new(d_model, d_ff).init_with(record.linear_1);
        let linear2 = LinearConfig::new(d_ff, d_model).init_with(record.linear_2);
        let dropout = DropoutConfig::new(dropout.unwrap_or(0.1)).init();
        Self {
            linear_1: linear1,
            linear_2: linear2,
            dropout,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.linear_1.forward(x);
        let x = relu(x);
        let x = self.dropout.forward(x);

        self.linear_2.forward(x)
    }
}

#[derive(Module, Debug)]
struct EncoderLayer<B: Backend> {
    norm_1: Norm<B>,
    norm_2: Norm<B>,
    attn: MultiHeadAttention<B>,
    ff: FeedForward<B>,
    dropout1: Dropout,
    dropout2: Dropout,
}

impl<B: Backend> EncoderLayer<B> {
    pub fn new_with(
        d_model: usize,
        heads: usize,
        dropout: Option<f64>,
        device: &B::Device,
        record: EncoderLayerRecord<B>,
    ) -> Self {
        let norm1 = Norm::new_with(d_model, None, device);
        let norm2 = Norm::new_with(d_model, None, device);
        let attn = MultiHeadAttention::new_with(d_model, heads, dropout, record.attn);
        let ff = FeedForward::new_with(d_model, None, dropout, record.ff);
        let dropout1 = DropoutConfig::new(dropout.unwrap_or(0.1)).init();
        let dropout2 = DropoutConfig::new(dropout.unwrap_or(0.1)).init();

        Self {
            norm_1: norm1,
            norm_2: norm2,
            attn,
            ff,
            dropout1,
            dropout2,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x2 = self.norm_1.forward(x.clone());
        let x = x.clone()
            + self
                .dropout1
                .forward(self.attn.forward(x2.clone(), x2.clone(), x2));

        let x2 = self.norm_2.forward(x.clone());

        x + self.dropout2.forward(self.ff.forward(x2))
    }
}

#[derive(Module, Debug)]
struct Encoder<B: Backend> {
    n: usize,
    pe: PositionalEncoder<B>,
    layers: Vec<EncoderLayer<B>>,
    norm: Norm<B>,
}

impl<B: Backend> Encoder<B> {
    pub fn new_with(
        hidden_size: usize,
        n: usize,
        heads: usize,
        dropout: Option<f64>,
        device: &B::Device,
        record: EncoderRecord<B>,
    ) -> Self {
        let pe = PositionalEncoder::new_with(hidden_size, None, dropout);
        let norm = Norm::new_with(hidden_size, None, &B::Device::default());
        let mut layers_record = record.layers;
        let layers = (0..n)
            .map(|_| {
                EncoderLayer::new_with(hidden_size, heads, dropout, device, layers_record.remove(0))
            })
            .collect();
        Self {
            n,
            pe,
            layers,
            norm,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut output = x;
        for layer in &self.layers {
            output = layer.forward(output);
        }

        output
    }
}

#[derive(Module, Debug)]
struct Decoder<B: Backend> {
    linear: SequentialLinear<B>,
}

impl<B: Backend> Decoder<B> {
    pub fn new_with(hidden_size: usize, output_size: usize, record: DecoderRecord<B>) -> Self {
        let mut linears_record = record.linear.layers;
        let linear = SequentialLinear::from(vec![
            LinearConfig::new(hidden_size, hidden_size).init_with(linears_record.remove(0)),
            LinearConfig::new(hidden_size, output_size).init_with(linears_record.remove(0)),
        ]);
        Self { linear }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.linear.forward(x);
        relu(x)
    }
}

#[derive(Module, Debug)]
struct Transformer<B: Backend> {
    linear: Linear<B>,
    encoder: Encoder<B>,
    decoder: Decoder<B>,
}

impl<B: Backend> Transformer<B> {
    pub fn new_with(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        n_layers: usize,
        heads: usize,
        dropout: Option<f64>,
        device: &B::Device,
        record: TransformerRecord<B>,
    ) -> Self {
        let linear = LinearConfig::new(input_size, hidden_size).init_with(record.linear);
        let encoder = Encoder::new_with(
            hidden_size,
            n_layers,
            heads,
            dropout,
            device,
            record.encoder,
        );
        let decoder = Decoder::new_with(hidden_size, output_size, record.decoder);
        Self {
            linear,
            encoder,
            decoder,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.linear.forward(x);
        let x = self.encoder.forward(x);
        self.decoder.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct LoreProcessModel<B: Backend> {
    stacker: Stacker<B>,
    tsfm_axis: Transformer<B>,
    x_position_embeddings: Embedding<B>,
    y_position_embeddings: Embedding<B>,
}

impl<B: Backend> LoreProcessModel<B> {
    pub fn new_with(record: LoreProcessModelRecord<B>, device: &B::Device) -> Self {
        let input_size = 256;
        let output_size = 4;
        let hidden_size = 256;
        let max_fmp_size = 256;
        let stacking_layers = 4;
        let tsfm_layers = 4;
        let num_heads = 8;
        let att_dropout = 0.1;

        let stacker = Stacker::new_with(
            output_size,
            hidden_size,
            output_size,
            stacking_layers,
            None,
            None,
            device,
            record.stacker,
        );
        let tsfm_axis = Transformer::new_with(
            input_size,
            hidden_size,
            output_size,
            tsfm_layers,
            num_heads,
            Some(att_dropout),
            device,
            record.tsfm_axis,
        );
        let x_position_embeddings =
            EmbeddingConfig::new(max_fmp_size, hidden_size).init_with(record.x_position_embeddings);
        let y_position_embeddings =
            EmbeddingConfig::new(max_fmp_size, hidden_size).init_with(record.y_position_embeddings);
        Self {
            stacker,
            tsfm_axis,
            x_position_embeddings,
            y_position_embeddings,
        }
    }

    pub fn forward(
        &self,
        outputs: Tensor<B, 3>,
        batch: Option<Tensor<B, 3>>,
        dets: Option<Tensor<B, 3, Int>>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let vis_feat = if let Some(batch) = &batch {
            batch.clone()
        } else {
            outputs
        };

        if batch.is_some() {
            panic!("batch 不为空是什么情况？")
        }

        if let Some(dets) = dets {
            let left_pe = self
                .x_position_embeddings
                .forward(Self::get_embedded_tensor(&dets, 0));
            let upper_pe = self
                .y_position_embeddings
                .forward(Self::get_embedded_tensor(&dets, 1));
            let right_pe = self
                .x_position_embeddings
                .forward(Self::get_embedded_tensor(&dets, 2));
            let lower_pe = self
                .y_position_embeddings
                .forward(Self::get_embedded_tensor(&dets, 5));
            let feat = vis_feat + left_pe + upper_pe + right_pe + lower_pe;

            let logic_axis = self.tsfm_axis.forward(feat.clone());
            let stacked_axis = self.stacker.forward(feat, logic_axis.clone());
            (logic_axis, stacked_axis)
        } else {
            let logic_axis = self.tsfm_axis.forward(vis_feat.clone());
            let stacked_axis = self.stacker.forward(vis_feat, logic_axis.clone());
            (logic_axis, stacked_axis)
        }
    }

    fn get_embedded_tensor(tensor: &Tensor<B, 3, Int>, d2: usize) -> Tensor<B, 2, Int> {
        let [bs, d1, _] = tensor.dims();
        tensor
            .clone()
            .slice([0..bs, 0..d1, 0..d2 + 1])
            .reshape([bs as i32, -1])
    }
}
