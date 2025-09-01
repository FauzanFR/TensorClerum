use ndarray::{s, stack, Array3, Array4, ArrayViewMut3, Axis};
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy)]
struct BoundingBox {
    min: [usize; 3],
    max: [usize; 3],
}

impl BoundingBox {
    fn new(min: [usize; 3], max: [usize; 3]) -> Self {
        Self { min, max }
    }

    fn intersects(&self, other: &BoundingBox) -> bool {
        self.min[0] <= other.max[0] && self.max[0] >= other.min[0] &&
        self.min[1] <= other.max[1] && self.max[1] >= other.min[1] &&
        self.min[2] <= other.max[2] && self.max[2] >= other.min[2]
    }

    fn contains(&self, other: &BoundingBox) -> bool {
        self.min[0] <= other.min[0] && self.max[0] >= other.max[0] &&
        self.min[1] <= other.min[1] && self.max[1] >= other.max[1] &&
        self.min[2] <= other.min[2] && self.max[2] >= other.max[2]
    }

    fn dimensions(&self) -> [usize; 3] {
        [
            self.max[0] - self.min[0] + 1,
            self.max[1] - self.min[1] + 1,
            self.max[2] - self.min[2] + 1,
        ]
    }
}

struct OctreeNode {
    bounds: BoundingBox,
    children: Option<[Box<OctreeNode>; 8]>,
    occupied: bool,
    data_index: Option<usize>,
}

impl OctreeNode {
    fn new(bounds: BoundingBox) -> Self {
        Self {
            bounds,
            children: None,
            occupied: false,
            data_index: None,
        }
    }

    fn subdivide(&mut self) {
        let [min_x, min_y, min_z] = self.bounds.min;
        let [max_x, max_y, max_z] = self.bounds.max;
        
        let mid_x = (min_x + max_x) / 2;
        let mid_y = (min_y + max_y) / 2;
        let mid_z = (min_z + max_z) / 2;

        self.children = Some([
            Box::new(OctreeNode::new(BoundingBox::new([min_x, min_y, min_z], [mid_x, mid_y, mid_z]))),
            Box::new(OctreeNode::new(BoundingBox::new([mid_x + 1, min_y, min_z], [max_x, mid_y, mid_z]))),
            Box::new(OctreeNode::new(BoundingBox::new([min_x, mid_y + 1, min_z], [mid_x, max_y, mid_z]))),
            Box::new(OctreeNode::new(BoundingBox::new([mid_x + 1, mid_y + 1, min_z], [max_x, max_y, mid_z]))),
            Box::new(OctreeNode::new(BoundingBox::new([min_x, min_y, mid_z + 1], [mid_x, mid_y, max_z]))),
            Box::new(OctreeNode::new(BoundingBox::new([mid_x + 1, min_y, mid_z + 1], [max_x, mid_y, max_z]))),
            Box::new(OctreeNode::new(BoundingBox::new([min_x, mid_y + 1, mid_z + 1], [mid_x, max_y, max_z]))),
            Box::new(OctreeNode::new(BoundingBox::new([mid_x + 1, mid_y + 1, mid_z + 1], [max_x, max_y, max_z]))),
        ]);
    }

    fn insert(&mut self, bounds: BoundingBox, data_index: usize) -> Option<BoundingBox> {
        if self.occupied || !self.bounds.contains(&bounds) {
            return None;
        }

        if let Some(children) = &mut self.children {
            for child in children.iter_mut() {
                if let Some(placement) = child.insert(bounds, data_index) {
                    return Some(placement);
                }
            }
            return None;
        }

        if self.bounds.contains(&bounds) {
            self.occupied = true;
            self.data_index = Some(data_index);
            return Some(self.bounds);
        }

        self.subdivide();
        self.insert(bounds, data_index)
    }
}

struct MemoryOrdo3 {
    x_max: usize,
    y_max: usize,
    z_max: usize,
    coords: Vec<(usize, usize, usize, usize, usize, usize, usize, usize)>,
    octree: Option<OctreeNode>,
}

impl MemoryOrdo3 {
    fn init(x_max: usize, y_max: usize, z_max: usize) -> Self {
        let root_bounds = BoundingBox::new([0, 0, 0], [x_max - 1, y_max - 1, z_max - 1]);
        
        Self {
            x_max,
            y_max,
            z_max,
            coords: Vec::new(),
            octree: Some(OctreeNode::new(root_bounds)),
        }
    }

    fn print(&self) {
        println!("x_max: {:?}", self.x_max);
        println!("y_max: {:?}", self.y_max);
        println!("z_max: {:?}", self.z_max);
        for (i, j, k, l, m, n, o, p) in &self.coords {
            println!("index: {} | {} ~ {}, {} ~ {}, {} ~ {} | layer: {}", i, j, k, l, m, n, o, p);
        }
    }

    fn make_layer(&self) -> Array3<f32> {
        Array3::<f32>::zeros((self.x_max, self.y_max, self.z_max))
    }

    fn get(&self, index: usize) -> Option<(usize, usize, usize, usize, usize, usize, usize)> {
        self.coords
            .iter()
            .find(|&&(idx, ..)| idx == index)
            .map(|&(_, x1, x2, y1, y2, z1, z2, h)| (x1, x2, y1, y2, z1, z2, h))
    }

    fn insert_array(&mut self, index: usize, shape: [usize; 3]) -> Option<BoundingBox> {
        let bounds = BoundingBox::new([0, 0, 0], [shape[0] - 1, shape[1] - 1, shape[2] - 1]);
        
        if let Some(octree) = &mut self.octree {
            if let Some(placement) = octree.insert(bounds, index) {
                let [min_x, min_y, min_z] = placement.min;
                let [max_x, max_y, max_z] = placement.max;
                
                self.coords.push((
                    index,
                    min_x, max_x,
                    min_y, max_y,
                    min_z, max_z,
                    self.coords.len()
                ));
                
                return Some(placement);
            }
        }
        
        None
    }
}

pub struct arr3_to_tensor {
    memory: MemoryOrdo3,
    data_array3: Vec<Array3<f32>>,
    data_array3_fix: Vec<Array3<f32>>,
    data_array4_fix: Array4<f32>,
    true_index: Vec<usize>,
}

impl arr3_to_tensor {
    pub fn init(array: Vec<Array3<f32>>) -> Self {
        let mut max_dims = [0, 0, 0];
        
        for arr in &array {
            let dims = arr.dim();
            max_dims[0] = max_dims[0].max(dims.0);
            max_dims[1] = max_dims[1].max(dims.1);
            max_dims[2] = max_dims[2].max(dims.2);
        }

        let memory = MemoryOrdo3::init(max_dims[0], max_dims[1], max_dims[2]);
        
        // Sort arrays by volume (largest first) for better packing
        let mut indexed: Vec<(usize, Array3<f32>)> = array.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| {
            let vol_a = a.1.dim().0 * a.1.dim().1 * a.1.dim().2;
            let vol_b = b.1.dim().0 * b.1.dim().1 * b.1.dim().2;
            vol_b.cmp(&vol_a)  // Descending order
        });

        let (true_index, data_array3) = indexed.into_iter().unzip();
        
        Self {
            memory,
            data_array3,
            data_array3_fix: Vec::new(),
            data_array4_fix: Array4::zeros((1, 1, 1, 1)),
            true_index,
        }
    }

    pub fn count(&mut self) {
        let first_shape = self.data_array3[0].shape();
        let all_equal = self.data_array3.iter().all(|a| a.shape() == first_shape);
        
        if all_equal {
            self.stacking()
        } else {
            self.packing()
        }
    }

    fn stacking(&mut self) {
        let mut layers: Vec<Array3<f32>> = Vec::new();
        
        for (idx, arr) in self.data_array3.iter().enumerate() {
            let mut layer = self.memory.make_layer();
            let shape = arr.dim();
            
            self.memory.coords.push((
                self.true_index[idx], 
                0, shape.0 - 1,
                0, shape.1 - 1,
                0, shape.2 - 1,
                idx
            ));
            
            layer.slice_mut(s![0..shape.0, 0..shape.1, 0..shape.2]).assign(arr);
            layers.push(layer);
        }
        
        self.data_array3_fix = layers;
        self.memory.print();
        self.data_array3 = Vec::new();
    }

    fn packing(&mut self) {
        let mut layers: Vec<Array3<f32>> = Vec::new();
        
        for (idx, arr) in self.data_array3.iter().enumerate() {
            let shape = arr.dim();
            let bounds = [shape.0, shape.1, shape.2];
            
            if let Some(placement) = self.memory.insert_array(self.true_index[idx], bounds) {
                let layer_idx = self.memory.coords.len() - 1;
                
                while layers.len() <= layer_idx {
                    layers.push(self.memory.make_layer());
                }
                
                let [min_x, min_y, min_z] = placement.min;
                layers[layer_idx].slice_mut(s![
                    min_x..=placement.max[0],
                    min_y..=placement.max[1],
                    min_z..=placement.max[2]
                ]).assign(arr);
            } else {
                let mut layer = self.memory.make_layer();
                layer.slice_mut(s![0..shape.0, 0..shape.1, 0..shape.2]).assign(arr);
                
                self.memory.coords.push((
                    self.true_index[idx], 
                    0, shape.0 - 1,
                    0, shape.1 - 1,
                    0, shape.2 - 1,
                    layers.len()
                ));
                
                layers.push(layer);
            }
        }
        
        self.data_array3_fix = layers;
        self.memory.print();
        self.data_array3 = Vec::new();
    }

    pub fn to_tensor4(&mut self) {
        let views: Vec<_> = self.data_array3_fix.iter().map(|a| a.view()).collect();
        self.data_array4_fix = stack(Axis(0), &views).unwrap();
        self.data_array3_fix = Vec::new();
    }

    pub fn get(&mut self, index: usize) -> ArrayViewMut3<f32> {
        let (x1, x2, y1, y2, z1, z2, h) = self.memory
            .get(index)
            .expect(&format!("Invalid index {} in arr3_to_tensor::get", index));
        self.data_array4_fix.slice_mut(s![h, x1..=x2, y1..=y2, z1..=z2])
    }

    pub fn dim(&self, index: usize) -> (usize, usize, usize) {
        let (x1, x2, y1, y2, z1, z2, _) = self.memory
            .get(index)
            .expect(&format!("Invalid index {} in arr3_to_tensor::dim", index));
        (x2 - x1 + 1, y2 - y1 + 1, z2 - z1 + 1)
    }

    pub fn len(&self) -> usize {
        self.memory.coords.len()
    }
}