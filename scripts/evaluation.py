def compute_metrics(gt_image, pred_image, loss_fn_alex):
    """Compute image quality metrics"""
    # Resize pred_image if needed
    if gt_image.shape != pred_image.shape:
        pred_image = cv2.resize(pred_image, (gt_image.shape[1], gt_image.shape[0]))

    # Convert to float32
    gt_float = gt_image.astype(np.float32) / 255.0
    pred_float = pred_image.astype(np.float32) / 255.0

    # PSNR
    psnr = compare_psnr(gt_float, pred_float, data_range=1.0)

    # SSIM
    ssim = compare_ssim(gt_float, pred_float, multichannel=True, data_range=1.0, win_size=3)
    
    # LPIPS - Move tensors to GPU
    gt_tensor = torch.tensor(gt_float).permute(2, 0, 1).unsqueeze(0).float().cuda()
    pred_tensor = torch.tensor(pred_float).permute(2, 0, 1).unsqueeze(0).float().cuda()
    lpips_value = loss_fn_alex(gt_tensor, pred_tensor).item()

    return psnr, ssim, lpips_value

def process_scene(scene_name):
    """Process a single scene"""
    print(f"\nProcessing scene: {scene_name}")
    
    # Set up paths for this scene
    input_path = os.path.join(BASE_INPUT_PATH, scene_name, "input")
    output_path = os.path.join(BASE_OUTPUT_PATH, scene_name, "output")
    
    # Initialize LPIPS and move to GPU
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()
    
    # Load reconstructed mesh
    mesh_path = os.path.join(input_path, "dense", "mesh.obj")
    if not os.path.exists(mesh_path):
        print(f"Error: Mesh file not found at {mesh_path}")
        return None
    
    mesh = trimesh.load(mesh_path)
    mesh = pyrender.Mesh.from_trimesh(mesh)
    
    # Load COLMAP cameras and images
    sparse_dir = os.path.join(input_path, "sparse", "0")
    if not os.path.exists(sparse_dir):
        print(f"Error: Sparse reconstruction directory not found at {sparse_dir}")
        return None
    
    print("Loading cameras and images...")
    cameras = read_cameras_binary(os.path.join(sparse_dir, "cameras.bin"))
    images = read_images_binary(os.path.join(sparse_dir, "images.bin"))
    
    # Create output directory for rendered images
    rendered_dir = os.path.join(output_path, "rendered")
    os.makedirs(rendered_dir, exist_ok=True)
    
    # Initialize metrics
    total_psnr = 0
    total_ssim = 0
    total_lpips = 0
    num_images = 0
    
    # Process each image
    for image_id, image in tqdm(images.items(), desc="Processing images"):
        # Load ground truth image
        gt_image_path = os.path.join(input_path, "images", image.name)
        if not os.path.exists(gt_image_path):
            print(f"Warning: Ground truth image not found at {gt_image_path}")
            continue
            
        gt_image = cv2.imread(gt_image_path)
        if gt_image is None:
            print(f"Warning: Could not read image at {gt_image_path}")
            continue
            
        # Get camera parameters
        camera = cameras[image.camera_id]
        K, width, height = load_intrinsics(camera)
        pose = load_extrinsics(image)
        
        # Render image
        rendered_image = render_scene(mesh, K, width, height, pose)
        
        # Save rendered image
        rendered_path = os.path.join(rendered_dir, f"rendered_{image.name}")
        cv2.imwrite(rendered_path, rendered_image)
        
        # Compute metrics
        psnr, ssim, lpips_value = compute_metrics(gt_image, rendered_image, loss_fn_alex)
        
        # Update totals
        total_psnr += psnr
        total_ssim += ssim
        total_lpips += lpips_value
        num_images += 1
        
        # Print per-image metrics
        print(f"\nImage: {image.name}")
        print(f"PSNR: {psnr:.2f}")
        print(f"SSIM: {ssim:.4f}")
        print(f"LPIPS: {lpips_value:.4f}")
    
    # Compute and save metrics
    if num_images > 0:
        avg_psnr = total_psnr / num_images
        avg_ssim = total_ssim / num_images
        avg_lpips = total_lpips / num_images
        
        print("\nAverage Metrics:")
        print(f"PSNR: {avg_psnr:.2f}")
        print(f"SSIM: {avg_ssim:.4f}")
        print(f"LPIPS: {avg_lpips:.4f}")
        
        # Save metrics to file
        metrics_file = os.path.join(output_path, "metrics.txt")
        with open(metrics_file, "w") as f:
            f.write(f"Average PSNR: {avg_psnr:.2f}\n")
            f.write(f"Average SSIM: {avg_ssim:.4f}\n")
            f.write(f"Average LPIPS: {avg_lpips:.4f}\n")
            f.write(f"Number of images processed: {num_images}\n")
        
        return {
            "scene": scene_name,
            "psnr": avg_psnr,
            "ssim": avg_ssim,
            "lpips": avg_lpips,
            "num_images": num_images
        }
    else:
        print("No images were processed successfully")
        return None 