use std::{io::Read};
use rand::prelude::*;
const TRAIN_PATH : &str = "./data/mnist/train-images.idx3-ubyte";
const TRAIN_LBL_PATH : &str = "./data/mnist/train-labels.idx1-ubyte";
const INPUT_SZ : i32 = 784;
const HIDDEN_SZ : i32 = 256;
const OUTPUT_SZ : i32 = 10;
const LEARNING_RATE : f32 = 0.001;
const REPS : i32 = 20;
const BATCH_SZ : i32 = 64;
const IMG_SZ : i32 = 28;
const TRAIN_SPL : f32 = 0.8;
const RAND_MAX : f32 = 2147483647.0;

struct Layer {
    weights : Vec<f32>,
    biases : Vec<f32>,
    input_sz : i32,
    output_sz : i32,
}
impl Layer {

    fn new(insize : i32, outsize : i32) -> Layer {
        let scale = (2.0/insize as f32).sqrt();
        let n = insize * outsize;
        let mut weights = vec![0.0; n as usize];
        for i in 0..n as usize {
            weights[i] = (rand::random::<f32>() * RAND_MAX - 0.5) * scale * 2.;
        }

        Layer {
            input_sz : insize,
            output_sz : outsize,
            biases: vec![0.0; outsize as usize], 
            weights
        }
    }

}
struct Network {
    hidden: Layer,
    output : Layer
}

struct InputData<'a> {
   images: &'a str,
   labels: &'a str,
   num_imgs: i32
}
fn softmax(input : Vec<f32>) -> Vec<f32> {
    let mut output: Vec<f32> = Vec::with_capacity(input.len());
    let max = input.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
    let mut sum = 0.0;

    for &item in &input {
        let res = (item - max).exp();
        output.push(res);
        sum += res;
    }

    for val in &mut output {
        *val /= sum;
    }

    output

}
fn read_labels() -> Vec<char> {
    let mut f = std::fs::File::open(TRAIN_LBL_PATH).expect("Training label path is bad");
    let mut buffer = [0u8; 8];
    f.read_exact(&mut buffer).expect("Failed to read header");

    let magic = i32::from_be_bytes(buffer[0..4].try_into().unwrap());
    let num_labels = i32::from_be_bytes(buffer[4..8].try_into().unwrap());
    println!("[{}] Number of labels: {}", magic, num_labels);

    let mut labels = Vec::with_capacity(num_labels as usize);
    for i in 0..num_labels {
        let mut label_byte = [0u8; 1];
        f.read_exact(&mut label_byte).expect("Failed to read label");
        let label = (label_byte[0] as u8 + b'0') as char;
        labels.push(label);
    }

    labels
}

fn read_images() -> Vec<char> {
    let mut f = std::fs::File::open(TRAIN_PATH).expect("Training image path is bad");
    let mut buffer = [0u8; 16];
    f.read_exact(&mut buffer).expect("Failure reading header bytes");
    let magic = i32::from_be_bytes(buffer[0..4].try_into().unwrap());
    let num_images = i32::from_be_bytes(buffer[4..8].try_into().unwrap());
    let num_rows = i32::from_be_bytes(buffer[8..12].try_into().unwrap());
    let num_cols = i32::from_be_bytes(buffer[12..16].try_into().unwrap());
    println!("[{}] Number of images: {}, Dimensions: {}x{}", magic, num_images, num_rows, num_cols);
    
    let mut ret: Vec<char> = Vec::with_capacity((num_images * num_rows * num_cols) as usize);
    for _ in 0..num_images {
        let mut image = vec![0u8; (num_rows * num_cols) as usize];
        f.read_exact(&mut image).expect("Failed to read a new image");
        for pixel in image {
            ret.push((pixel as char).to_ascii_lowercase());
        }
    }
    ret
}

/*
 * void shuffle_data(unsigned char *images, unsigned char *labels, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        for (int k = 0; k < INPUT_SIZE; k++) {
            unsigned char temp = images[i * INPUT_SIZE + k];
            images[i * INPUT_SIZE + k] = images[j * INPUT_SIZE + k];
            images[j * INPUT_SIZE + k] = temp;
        }
        unsigned char temp = labels[i];
        labels[i] = labels[j];
        labels[j] = temp;
    }
}
 * */
fn shuffle_data(imgs : &Vec<char>, labels : &Vec<char>, n : i32) 
    -> (Vec<char> , Vec<char>) 
    {
    let mut counter = n-1;
    let top = counter;
    let mut out_img : Vec<char> = vec!['~'; imgs.len()];
    let mut out_label : Vec<char> = vec!['~'; labels.len()];
    let mut rng = rand::thread_rng();

    while counter > 0 {

        let rand : i32 = rng.gen_range(1..top);
        for k in 0..INPUT_SZ {
            out_img[(rand * INPUT_SZ + k) as usize] = imgs[(counter * INPUT_SZ + k) as usize];
            out_img[(counter * INPUT_SZ + k) as usize] = imgs[(rand * INPUT_SZ + k) as usize ];

        }
        out_label[counter as usize] = labels[rand as usize];
        out_label[rand as usize] = labels[counter as usize];
        counter-=1;
    }

    (out_img, out_label)

}



fn main() {
    let start_time = std::time::Instant::now();

    let labels = read_labels();
    let images = read_images();
    let mut net = Network {
        hidden : Layer::new(INPUT_SZ, HIDDEN_SZ),
        output : Layer::new(HIDDEN_SZ, OUTPUT_SZ)
    };
    let num_imgs = images.len() as i32 / (IMG_SZ * IMG_SZ);
    let (shuffled_imgs, shuffled_labels) = shuffle_data(&images, &labels, num_imgs);

    let train_size = (TRAIN_SPL * images.len() as f32) as i32;
    let test_size = images.len() - train_size as usize;
    
    let duration = start_time.elapsed();
    println!("Setup time: {:?}", duration);
    // Now we actually train
}
