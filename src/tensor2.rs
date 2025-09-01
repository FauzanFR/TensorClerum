use std::fs::File;
use bincode;

use ndarray::{s, stack, Array2, Array3, ArrayViewMut2, Axis};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
struct ckpt_tensor3 {
    array: Array3<f32>,
    coords: Vec<(usize, usize, usize, usize, usize, usize)>,
    x_max: usize,
    y_max: usize,
}

fn save_checkpoint(path: &str, clr: &ckpt_tensor3) -> std::io::Result<()> {
    let file = File::create(path)?;
    bincode::serialize_into(file, clr).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
}

fn load_checkpoint(path: &str) -> std::io::Result<ckpt_tensor3> {
    let file = File::open(path)?;
    bincode::deserialize_from(file).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Rect {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
}

impl Rect {
    fn area(&self) -> usize {
        self.w * self.h
    }
    
    fn contains(&self, other: &Rect) -> bool {
        self.x <= other.x &&
        self.y <= other.y &&
        self.x + self.w >= other.x + other.w &&
        self.y + self.h >= other.y + other.h
    }
    
    fn overlap(&self, other: &Rect) -> bool {
        self.x < other.x + other.w &&
        self.x + self.w > other.x &&
        self.y < other.y + other.h &&
        self.y + self.h > other.y
    }
}

struct MaxRectsBin {
    width: usize,
    height: usize,
    free_rects: Vec<Rect>,
    used_rects: Vec<Rect>,
}

impl MaxRectsBin {
    
    fn new(width: usize, height: usize) -> Self {
        let mut free_rects = Vec::new();
        free_rects.push(Rect { x: 0, y: 0, w: width, h: height });
        
        Self {
            width,
            height,
            free_rects,
            used_rects: Vec::new(),
        }
    }
    
    fn insert(&mut self, rect: &Rect) -> Option<(usize, usize)> {
        let mut best_node = None;
        let mut best_short_side_fit = std::usize::MAX;

        for (i, free_rect) in self.free_rects.iter().enumerate().rev() {
            if free_rect.w >= rect.w && free_rect.h >= rect.h {
                let leftover_h = free_rect.h - rect.h;
                let leftover_w = free_rect.w - rect.w;
                let short_side = leftover_h.min(leftover_w);
                
                if short_side < best_short_side_fit {
                    best_node = Some((free_rect.x, free_rect.y, i));
                    best_short_side_fit = short_side;
                }
            }
        }

        if let Some((x, y, free_rect_index)) = best_node {
            let placed_rect = Rect { x, y, w: rect.w, h: rect.h };
            self.used_rects.push(placed_rect);
            
            let free_rect = self.free_rects.remove(free_rect_index);
            self.split_free_rect(&free_rect, &placed_rect);
            self.merge_free_rects();
            
            Some((x, y))
        } else {
            None
        }
    }
    
    fn split_free_rect(&mut self, free_rect: &Rect, placed_rect: &Rect) {
        if placed_rect.y > free_rect.y {
            let new_rect = Rect {
                x: free_rect.x,
                y: free_rect.y,
                w: free_rect.w,
                h: placed_rect.y - free_rect.y,
            };
            if new_rect.w > 0 && new_rect.h > 0 {
                self.free_rects.push(new_rect);
            }
        }
        
        if placed_rect.y + placed_rect.h < free_rect.y + free_rect.h {
            let new_rect = Rect {
                x: free_rect.x,
                y: placed_rect.y + placed_rect.h,
                w: free_rect.w,
                h: free_rect.y + free_rect.h - (placed_rect.y + placed_rect.h),
            };
            if new_rect.w > 0 && new_rect.h > 0 {
                self.free_rects.push(new_rect);
            }
        }
        
        if placed_rect.x > free_rect.x {
            let new_rect = Rect {
                x: free_rect.x,
                y: free_rect.y,
                w: placed_rect.x - free_rect.x,
                h: free_rect.h,
            };
            if new_rect.w > 0 && new_rect.h > 0 {
                self.free_rects.push(new_rect);
            }
        }
        
        if placed_rect.x + placed_rect.w < free_rect.x + free_rect.w {
            let new_rect = Rect {
                x: placed_rect.x + placed_rect.w,
                y: free_rect.y,
                w: free_rect.x + free_rect.w - (placed_rect.x + placed_rect.w),
                h: free_rect.h,
            };
            if new_rect.w > 0 && new_rect.h > 0 {
                self.free_rects.push(new_rect);
            }
        }
    }
    
    fn merge_free_rects(&mut self) {
        let mut i = 0;
        while i < self.free_rects.len() {
            let mut j = i + 1;
            while j < self.free_rects.len() {
                if self.try_merge_rects(i, j) {
                    self.free_rects.remove(j);
                    i = 0;
                    break;
                }
                j += 1;
            }
            i += 1;
        }
    }
    
    fn try_merge_rects(&mut self, i: usize, j: usize) -> bool {
        let a = self.free_rects[i];
        let b = self.free_rects[j];
        
        if a.y == b.y && a.h == b.h {
            if a.x + a.w == b.x {
                self.free_rects[i] = Rect { x: a.x, y: a.y, w: a.w + b.w, h: a.h };
                return true;
            }
            if b.x + b.w == a.x {
                self.free_rects[i] = Rect { x: b.x, y: b.y, w: a.w + b.w, h: a.h };
                return true;
            }
        }
        
        if a.x == b.x && a.w == b.w {
            if a.y + a.h == b.y {
                self.free_rects[i] = Rect { x: a.x, y: a.y, w: a.w, h: a.h + b.h };
                return true;
            }
            if b.y + b.h == a.y {
                self.free_rects[i] = Rect { x: a.x, y: b.y, w: a.w, h: a.h + b.h };
                return true;
            }
        }
        
        false
    }
}

struct MemoryOrdo2 {
    x_max: usize,
    y_max: usize,
    coords: Vec<(usize, usize, usize, usize, usize, usize)>,
}

impl MemoryOrdo2 {
    fn init() -> Self {
        Self {
            x_max: 0,
            y_max: 0,
            coords: Vec::new(),
        }
    }

    fn get(&self, index: usize) -> Option<(usize, usize, usize, usize, usize)> {
        self.coords
            .iter()
            .find(|&&(idx, ..)| idx == index)
            .map(|&(_, x1, x2, y1, y2, z)| (x1, x2, y1, y2, z))
    }

    fn print(&self) {
        println!("x_max: {:?}", self.x_max);
        println!("y_max: {:?}", self.y_max);
        for (i, j, k, l, m, n) in &self.coords {
            println!("index: {} | {} ~ {}, {} ~ {} | layer: {}", i, j, k, l, m, n);
        }
    }

    fn make_layer(&self) -> Array2<f32> {
        Array2::<f32>::zeros((self.x_max, self.y_max))
    }

    fn make_max_rects_bin(&self) -> MaxRectsBin {
        MaxRectsBin::new(self.x_max, self.y_max)
    }
}

pub struct arr2_to_tensor {
    memory: MemoryOrdo2,
    data_array2: Vec<Array2<f32>>,
    data_array2_fix: Vec<Array2<f32>>,
    data_array3_fix: Array3<f32>,
    true_index: Vec<usize>,
}

impl arr2_to_tensor {
    pub fn init(array: Vec<Array2<f32>>) -> Self {
        let mut memory = MemoryOrdo2::init();
        let mut indexed: Vec<(usize, Array2<f32>)> = array
            .into_iter()
            .enumerate()
            .map(|(i, arr)| {
                let (x, y) = (arr.shape()[0], arr.shape()[1]);
                memory.x_max = memory.x_max.max(x);
                memory.y_max = memory.y_max.max(y);
                (i, arr)
            })
            .collect();

        indexed.sort_by_key(|(_, v)| -(v.shape()[0] as isize * v.shape()[1] as isize));
        let (true_index, data_array2) = indexed.into_iter().unzip();

        Self {
            memory,
            data_array2,
            data_array2_fix: Vec::new(),
            data_array3_fix: Array3::zeros((1, 1, 1)),
            true_index,
        }
    }

    pub fn count(&mut self) {
        let first_shape = self.data_array2[0].shape();
        let all_equal = self.data_array2.iter().all(|a| a.shape() == first_shape);
        if all_equal{self.stacking()} else {self.packing();}
    }
    
    fn stacking(&mut self){
        let mut layers: Vec<Array2<f32>> = Vec::new();
        let remaining: Vec<(usize, &Array2<f32>)> = self.data_array2.iter().enumerate().collect();

        for &(idx, arr) in &remaining {
            let mut layer = self.memory.make_layer();
            self.memory.coords.push((self.true_index[idx], 0, 0 + self.memory.x_max - 1, 0, 0 + self.memory.y_max - 1, idx));
            layer.assign(arr);
            layers.push(layer);
        }
        self.data_array2_fix = layers;
        self.memory.print();
        self.data_array2 = Vec::new();
    }

    fn packing(&mut self){
        
        let mut remaining: Vec<(usize, &Array2<f32>)> = self.data_array2.iter().enumerate().collect();
        let mut layers: Vec<Array2<f32>> = Vec::new();
        let mut max_rects_bins: Vec<MaxRectsBin> = Vec::new();

        while !remaining.is_empty() {
            let mut placed_idx = Vec::new();

            for &(idx, arr) in &remaining {
                let (n1, n2) = arr.dim();
                let rect = Rect { x: 0, y: 0, w: n2, h: n1 };
                let mut placed = false;

                for (layer_id, (layer, bin)) in layers.iter_mut().zip(max_rects_bins.iter_mut()).enumerate() {
                    if let Some((x, y)) = bin.insert(&rect) {
                        layer.slice_mut(s![y..y + n1, x..x + n2]).assign(arr);
                        self.memory.coords.push((self.true_index[idx], x, x + n2 - 1, y, y + n1 - 1, layer_id));
                        placed_idx.push(idx);
                        placed = true;
                        break;
                    }
                }

                if !placed {
                    let mut new_layer = self.memory.make_layer();
                    let mut new_bin = self.memory.make_max_rects_bin();
                    
                    if let Some((x, y)) = new_bin.insert(&rect) {
                        new_layer.slice_mut(s![y..y + n1, x..x + n2]).assign(arr);
                        self.memory.coords.push((self.true_index[idx], x, x + n2 - 1, y, y + n1 - 1, layers.len()));
                        layers.push(new_layer);
                        max_rects_bins.push(new_bin);
                        placed_idx.push(idx);
                    } else {
                        panic!("Failed to insert into empty bin");
                    }
                }
            }

            remaining.retain(|(idx, _)| !placed_idx.contains(idx));
        }

        self.memory.print();
        self.data_array2_fix = layers;
        self.data_array2 = Vec::new();
    }

    pub fn to_tensor3(&mut self) {
        let views: Vec<_> = self.data_array2_fix.iter().map(|a| a.view()).collect();
        self.data_array3_fix = stack(Axis(0), &views).unwrap();
        self.data_array2_fix = Vec::new();
    }

    pub fn get(&mut self, index: usize) -> ArrayViewMut2<f32> {
        let (x1, x2, y1, y2, z) = self.memory
            .get(index)
            .expect(&format!("Invalid index {} in arr2_to_tensor::get", index));
        self.data_array3_fix.slice_mut(s![z, x1..=x2, y1..=y2])
    }

    pub fn dim (&mut self, index: usize) -> (usize, usize) {
        let (x1, x2, y1, y2, _) = self.memory
            .get(index)
            .expect(&format!("Invalid index {} in arr2_to_tensor::dim", index));
        (x2-x1+1, y2-y1+1)
    }
    
    pub fn len(&self) -> usize {
        self.memory.coords.len()
    }

    pub fn save(&self, path: &str) {
        let _ = save_checkpoint(path, &ckpt_tensor3 {
            array: self.data_array3_fix.clone(),
            coords: self.memory.coords.clone(),
            x_max: self.memory.x_max,
            y_max: self.memory.y_max,
        });
    }

    pub fn load(path: &str) -> Self {
        let arr = load_checkpoint(path).expect("Failed to load checkpoint");
        let mut mem = MemoryOrdo2::init();
        mem.coords = arr.coords;
        mem.x_max = arr.x_max;
        mem.y_max = arr.y_max;
        Self {
            memory: mem,
            data_array2: Vec::new(),
            data_array2_fix: Vec::new(),
            data_array3_fix: arr.array,
            true_index: Vec::new(),
        }
    }
}