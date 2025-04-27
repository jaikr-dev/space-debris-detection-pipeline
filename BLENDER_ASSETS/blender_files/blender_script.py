"""
Space Debris Detection - Blender Dataset Generator

This script automates the rendering of space debris objects in Blender for YOLO dataset generation.
It creates images with controlled randomization of object positions and orientations, generates
YOLO-format annotations, and organizes outputs into train/validation/test splits for machine learning.

Code base source: https://blender.stackexchange.com/a/7203
"""

#------------------------------------------------------------------------------
# Standard library imports
#------------------------------------------------------------------------------
import bpy                # Blender Python API for scene manipulation
import math               # Mathematical functions and constants
import random             # For generating random numbers
import time               # For measuring elapsed time
from mathutils import Euler, Quaternion  # For handling rotations
from pathlib import Path  # For platform-independent path operations
import sys                # For redirecting stdout/stderr for logging
import json               # For reading and writing JSON state files
import os                 # For file and directory operations

#------------------------------------------------------------------------------
# User Configuration Parameters
#------------------------------------------------------------------------------
# Dataset size configuration
TOTAL_IMAGES_PER_DEBRIS = 50    # Number of images to render per debris
MAX_BATCHES = 6                 # Maximum number of batches to process
TRAIN_SPLIT = 0.8               # Proportion of images for training
VAL_SPLIT = 0.1                 # Proportion of images for validation
TEST_SPLIT = 0.1                # Proportion of images for testing

# Debris objects to render
DEBRIS_NAMES = ['Satellite', 'Envisat', 'Hubble', 'Falcon 9 F&S']

# Output directory configuration
PROJECT_BASE = r"C:\Users\Jai\Documents\GitHub\space-debris-detection-pipeline"
ML_ASSETS_BASE = os.path.join(PROJECT_BASE, "ML_ASSETS")
DATASET_BASE = os.path.join(ML_ASSETS_BASE, "dataset")

# File paths for state management
STATE_FILE = bpy.path.abspath("//render_state.json")
FLAG_FILE = bpy.path.abspath("//stop_flag.txt")

# Batch processing limits
BATCH_LIMIT = TOTAL_IMAGES_PER_DEBRIS * len(DEBRIS_NAMES)  # Images per batch

#------------------------------------------------------------------------------
# Helper Functions for Geometry and Transformations
#------------------------------------------------------------------------------
def clamp(value, min_value, max_value):
    """
    Clamp a value to ensure it stays within the specified minimum and maximum bounds.
    
    Args:
        value (float): The value to clamp
        min_value (float): The minimum allowed value
        max_value (float): The maximum allowed value
    
    Returns:
        float: The clamped value
    """
    if value < min_value:
        return min_value
    elif value > max_value:
        return max_value
    else:
        return value

def get_2d_bounding_box(scene, camera_obj, target_obj):
    """
    Compute the 2D bounding box of a 3D object as seen from a specified camera.

    The function projects the object's vertices into camera space, normalizes them based on the 
    camera's view frame, and calculates the bounding box in pixel coordinates.

    Args:
        scene (bpy.types.Scene): The current Blender scene
        camera_obj (bpy.types.Object): The camera from which the object is viewed
        target_obj (bpy.types.Object): The object for which to calculate the bounding box

    Returns:
        tuple: A tuple containing:
               (Left X coordinate, Top Y coordinate, Center X coordinate, Center Y coordinate, Width, Height)
               in pixel units, or (0, 0, 0, 0) if the object is not visible
    """
    # Invert the normalized camera matrix to transform world coordinates to camera space
    camera_matrix = camera_obj.matrix_world.normalized().inverted()

    # Use the dependency graph to get the evaluated state of the target object (with modifiers applied)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated_target = target_obj.evaluated_get(depsgraph)

    # Create a temporary mesh copy and transform its vertices to world coordinates
    mesh_data = evaluated_target.to_mesh()
    mesh_data.transform(target_obj.matrix_world)

    # Transform the mesh vertices into camera (local) space
    mesh_data.transform(camera_matrix)

    # Retrieve camera data and calculate its view frame
    camera_data = camera_obj.data
    camera_view_frame = [-v for v in camera_data.view_frame(scene=scene)[:3]]  # Use first three corners

    # Determine if the camera is in perspective mode
    is_perspective = camera_data.type != 'ORTHO'

    # Lists to hold the normalized x and y coordinates for each vertex
    projected_x_coords = []
    projected_y_coords = []

    # Process each vertex to project it onto the camera's view
    for vertex in mesh_data.vertices:
        vertex_pos = vertex.co
        depth = -vertex_pos.z  # Depth is measured as the negative z value

        if is_perspective:
            if depth == 0.0:
                # For vertices exactly at the camera plane, default to center
                projected_x_coords.append(0.5)
                projected_y_coords.append(0.5)
            else:
                # Adjust view frame for perspective distortion
                camera_view_frame = [(v / (v.z / depth)) for v in camera_view_frame]

        # Define view frame boundaries
        left_bound, right_bound = camera_view_frame[1].x, camera_view_frame[2].x
        bottom_bound, top_bound = camera_view_frame[0].y, camera_view_frame[1].y

        # Normalize vertex coordinates within the view frame
        normalized_x = (vertex_pos.x - left_bound) / (right_bound - left_bound)
        normalized_y = (vertex_pos.y - bottom_bound) / (top_bound - bottom_bound)

        projected_x_coords.append(normalized_x)
        projected_y_coords.append(normalized_y)

    # Clamp the normalized coordinates to [0, 1] range
    min_x = clamp(min(projected_x_coords), 0.0, 1.0)
    max_x = clamp(max(projected_x_coords), 0.0, 1.0)
    min_y = clamp(min(projected_y_coords), 0.0, 1.0)
    max_y = clamp(max(projected_y_coords), 0.0, 1.0)

    # Free the temporary mesh data
    evaluated_target.to_mesh_clear()

    # Get render resolution settings
    render_settings = scene.render
    resolution_scale = render_settings.resolution_percentage * 0.01
    render_width = render_settings.resolution_x * resolution_scale
    render_height = render_settings.resolution_y * resolution_scale

    # If the bounding box has zero area, return zeros
    if round((max_x - min_x) * render_width) == 0 or round((max_y - min_y) * render_height) == 0:
        return (0, 0, 0, 0)

    # Convert normalized values to pixel coordinates
    return (
        round((min_x * render_width), 6),                                               # Left X coordinate
        round((render_height - max_y * render_height), 6),                               # Top Y coordinate (inverted Y axis)
        round(((min_x * render_width) + ((max_x - min_x) * render_width) / 2), 6),       # Center X coordinate
        round(((render_height - max_y * render_height) + ((max_y - min_y) * render_height) / 2), 6),  # Center Y coordinate
        round(((max_x - min_x) * render_width), 6),                                      # Width in pixels
        round(((max_y - min_y) * render_height), 6)                                      # Height in pixels
    )

#------------------------------------------------------------------------------
# Object Transformation and Scene Manipulation Functions
#------------------------------------------------------------------------------
def reset_axis_euler(axis_to_change):
    """
    Reset an object's Euler rotation angles to zero.

    Args:
        axis_to_change (bpy.types.Object): The object whose rotation will be reset
    """
    axis_to_change.rotation_mode = 'XYZ'
    axis_to_change.rotation_euler = Euler((0, 0, 0), 'XYZ')

def randomly_rotate_axis_euler(obj, axes=('Z',)):
    """
    Apply a random rotation to the specified Euler axes of an object.

    Args:
        obj (bpy.types.Object): The object to rotate
        axes (tuple of str): Axes to randomize ('X', 'Y', 'Z'). Default rotates around 'Z' only

    Returns:
        list: The new Euler angles as a list
    """
    obj.rotation_mode = 'XYZ'
    current_angles = list(obj.rotation_euler)
    axis_index = {'X': 0, 'Y': 1, 'Z': 2}

    for ax in axes:
        idx = axis_index[ax.upper()]
        current_angles[idx] = random.random() * 2 * math.pi

    obj.rotation_euler = Euler(tuple(current_angles), 'XYZ')
    return current_angles

def randomly_rotate_object_quaternion(object_to_change):
    """
    Apply a random 3D rotation to an object using a quaternion.

    Args:
        object_to_change (bpy.types.Object): The object to rotate

    Returns:
        Quaternion: The random quaternion applied to the object
    """
    # Generate three random numbers for quaternion components
    u1, u2, u3 = random.random(), random.random(), random.random()

    # Use Shoemake's algorithm to compute a random unit quaternion
    qx = math.sqrt(1 - u1) * math.sin(2 * math.pi * u2)
    qy = math.sqrt(1 - u1) * math.cos(2 * math.pi * u2)
    qz = math.sqrt(u1) * math.sin(2 * math.pi * u3)
    qw = math.sqrt(u1) * math.cos(2 * math.pi * u3)

    quaternion = Quaternion((qw, qx, qy, qz))

    # Set the object's rotation mode and apply the quaternion
    object_to_change.rotation_mode = 'QUATERNION'
    object_to_change.rotation_quaternion = quaternion
    return quaternion

def randomly_set_container_offset():
    """
    Randomly adjust the 'Follow Path' constraint offsets for specified container objects.
    
    This affects the circular movement of objects like debris, debris circle container, and camera container.
    """
    bpy.context.scene.objects['Debris_Tracking_Container'].constraints['Follow Path'].offset = random.random() * 100
    bpy.context.scene.objects['Orbit_Container'].constraints['Follow Path'].offset = random.random() * 100
    bpy.context.scene.objects['Camera_Container'].constraints['Follow Path'].offset = random.random() * 100

#------------------------------------------------------------------------------
# Scene Object and State Management Functions
#------------------------------------------------------------------------------
def check_object_in_scene(object_name):
    """
    Check if an object with the specified name exists in the current scene.

    Args:
        object_name (str): Name of the object to search for

    Returns:
        bpy.types.Object or None: The found object or None if not present
    """
    if object_name in bpy.context.scene.objects:
        obj = bpy.context.scene.objects[object_name]
        print(f"Object '{object_name}' found: {obj}")
        return obj
    else:
        print(f"Object '{object_name}' not found in the scene.")
        return None

def load_or_initialize_state():
    """
    Load the rendering state from a JSON file or initialize it if the file does not exist.

    Returns:
        tuple: A tuple containing the state dictionary and the path to the state file
    """
    state_file = STATE_FILE
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            state = json.load(f)
    else:
        state = {"batch": 1, "start_idx": 0, "no_of_batches": 1}

    # Ensure the state dictionary has all required keys
    if "batch" not in state:
        state["batch"] = 1
    if "start_idx" not in state:
        state["start_idx"] = 0
    if "no_of_batches" not in state:
        state["no_of_batches"] = 0

    return state, state_file

def assign_classes_to_debris(debris_names):
    """
    Assign class IDs to debris objects based on their order in the provided list.

    Args:
        debris_names (list of str): List of debris object names
    """
    for class_num, debris_name in enumerate(debris_names):
        try:
            obj = bpy.context.scene.objects[debris_name]
            obj["Class"] = class_num
            print(f"Set Class = {class_num} for {debris_name}")
        except KeyError:
            print(f"Object '{debris_name}' not found in the scene.")

#------------------------------------------------------------------------------
# Dataset Generation and File Management Functions
#------------------------------------------------------------------------------
def create_config_file(output_path, debris_names, batch_number):
    """
    Create a YAML configuration file for the dataset.

    Args:
        output_path (Path): Directory where the config file will be created
        debris_names (list of str): List of debris names (used for class names)
        batch_number (int): The current batch number for naming purposes
    """
    # Ensure the output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    config_file = output_path / f"config.yaml"

    # Create the configuration content with forward slashes
    config_data = f"""path: {output_path.as_posix()}
train: train/images
val: val/images
test: test/images

names:
  0: {debris_names[0]}
  1: {debris_names[1]}
  2: {debris_names[2]}
  3: {debris_names[3]}
"""

    with open(config_file, "w") as f:
        f.write(config_data)

def save_state_and_exit(state, state_file, batch_limit_reached=False):
    """
    Save the current state to file and exit Blender.

    Args:
        state (dict): The state dictionary to be saved
        state_file (str): Path to the JSON state file
        batch_limit_reached (bool): If True, a stop flag file will be created
    """
    bpy.ops.wm.save_mainfile()  # Save the current Blender file
    with open(state_file, "w") as f:
        json.dump(state, f)

    if batch_limit_reached:
        flag_file = FLAG_FILE
        with open(flag_file, "w") as f:
            f.write("Batch number limit reached.")

    print("Exiting Blender...")
    bpy.ops.wm.quit_blender()

#------------------------------------------------------------------------------
# Main Dataset Rendering Function
#------------------------------------------------------------------------------
def render_dataset(state, state_file, debris_names, total_images, axes, clouds_obj, sun_obj):
    """
    Render dataset images and generate YOLO-format annotations for each debris object.

    Args:
        state (dict): The current rendering state
        state_file (str): Path to the JSON state file
        debris_names (list of str): Names of debris objects to render
        total_images (int): Number of images to render per debris
        axes (list): List of objects (axes) that may be randomly rotated
        clouds_obj (bpy.types.Object): The clouds object (optional rotation)
        sun_obj (bpy.types.Object): The sun object (optional rotation)
    """
    current_batch = state["batch"]
    current_start_idx = state["start_idx"]
    current_no_of_batches = state["no_of_batches"]

    # Disable persistent data to avoid memory issues during batch rendering
    bpy.context.scene.cycles.use_persistent_data = False

    if current_no_of_batches > MAX_BATCHES:
        print(f"Batches completed: {current_no_of_batches}, limit is {MAX_BATCHES}.")
        save_state_and_exit(state, state_file, batch_limit_reached=True)
        return

    #--------------------------------------------------------------------------
    # Define dataset splits based on configuration parameters
    #--------------------------------------------------------------------------
    debris_renders_per_split = [
        ('train', int(total_images * TRAIN_SPLIT)),  # Training split
        ('val', int(total_images * VAL_SPLIT)),      # Validation split 
        ('test', int(total_images * TEST_SPLIT))     # Testing split
    ]

    #--------------------------------------------------------------------------
    # Set up output directories
    #--------------------------------------------------------------------------
    output_path = Path(DATASET_BASE) / f"Debris{current_batch}"
    create_config_file(output_path, debris_names, current_batch)

    # Calculate the total number of renders for the current batch
    debris_count = len(debris_names)
    total_render_count = sum([debris_count * r[1] for r in debris_renders_per_split])
    print(f"Total Render Count: {total_render_count}")

    # Hide all debris objects initially
    for name in debris_names:
        if name != 'Empty':
            bpy.context.scene.objects[name].hide_render = True

    start_time = time.time()   # Start timer for the rendering process
    start_index_local = 0      # Local counter for batch progress

    #--------------------------------------------------------------------------
    # Process each dataset split (train/val/test)
    #--------------------------------------------------------------------------
    for split_name, renders_per_debris in debris_renders_per_split:
        tasks = []
        # Create render tasks for each debris object
        for debris in debris_names:
            tasks.extend([debris] * renders_per_debris)
        random.shuffle(tasks)  # Randomize the order of render tasks

        print(f"Starting split: {split_name} | Total renders for this split: {len(tasks)}")
        print("=============================================")

        # Define output directories for images, labels, and additional coordinate data
        output_dir_image = output_path / split_name / "images"
        output_dir_label = output_path / split_name / "labels"
        output_dir_coords = Path(DATASET_BASE) / "orbit_container_coordinates"
        output_dir_debris_tracking_container_coords = Path(DATASET_BASE) / "debris_tracking_container_coordinates"
        output_dir_camera_coords = Path(DATASET_BASE) / "camera_container_coordinates"
        output_dir_debris_orientation_euler = Path(DATASET_BASE) / "debris_orientation_euler"
        output_dir_debris_orientation_quaternion = Path(DATASET_BASE) / "debris_orientation_quaternion"

        for directory in [output_dir_image, output_dir_label, output_dir_coords,
                          output_dir_debris_tracking_container_coords, output_dir_camera_coords,
                          output_dir_debris_orientation_euler, output_dir_debris_orientation_quaternion]:
            directory.mkdir(parents=True, exist_ok=True)

        #----------------------------------------------------------------------
        # Render each object and generate annotations
        #----------------------------------------------------------------------
        for debris_name in tasks:
            print(f"Rendering: {split_name}/{debris_name} with index {str(current_start_idx).zfill(6)}")

            # Hide all debris objects, then unhide the current one
            for d in debris_names:
                bpy.context.scene.objects[d].hide_render = True
            debris_to_render = bpy.context.scene.objects[debris_name]
            debris_to_render.hide_render = False

            # Apply random rotations using quaternion and Euler methods
            q = None
            for axis in axes:
                if axis:
                    q = randomly_rotate_object_quaternion(axis)

            if clouds_obj:
                randomly_rotate_axis_euler(clouds_obj)
            if sun_obj:
                randomly_rotate_axis_euler(sun_obj)
            randomly_set_container_offset()

            # Render the image
            image_filepath = output_dir_image / f"{str(current_start_idx).zfill(6)}.png"
            bpy.context.scene.render.filepath = str(image_filepath)
            bpy.ops.render.render(write_still=True)

            # Get the class ID assigned to the debris object
            obj = bpy.data.objects.get(debris_name)
            object_class = obj["Class"] if obj and "Class" in obj else -1

            # Compute the 2D bounding box for annotation
            scene = bpy.context.scene
            camera = bpy.context.scene.objects['Camera']
            selected_object = bpy.context.scene.objects[debris_name]
            bounding_box = get_2d_bounding_box(scene, camera, selected_object)

            # Normalize bounding box coordinates relative to the image dimensions
            image_width = bpy.context.scene.render.resolution_x
            image_height = bpy.context.scene.render.resolution_y
            x_center = bounding_box[2] / image_width
            y_center = bounding_box[3] / image_height
            width = bounding_box[4] / image_width
            height = bounding_box[5] / image_height

            # Write YOLO-format annotation
            annotation_filepath = output_dir_label / f"{str(current_start_idx).zfill(6)}.txt"
            with open(annotation_filepath, "w") as f:
                annotation_line = f"{object_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                f.write(annotation_line + "\n")

            # Save global position coordinates
            world_position = selected_object.matrix_world.translation
            coords_filepath = output_dir_coords / f"{split_name}_{str(current_start_idx).zfill(6)}.txt"
            with open(coords_filepath, "w") as f:
                coords_line = f"{world_position.x:.6f} {world_position.y:.6f} {world_position.z:.6f}"
                f.write(coords_line + "\n")

            # Save Debris_Tracking_Container offset value
            container_offset = bpy.context.scene.objects["Debris_Tracking_Container"].constraints["Follow Path"].offset
            debris_tracking_container_coords_filepath = output_dir_debris_tracking_container_coords / f"{split_name}_{str(current_start_idx).zfill(6)}.txt"
            with open(debris_tracking_container_coords_filepath, "w") as f:
                f.write(f"{container_offset:.6f}\n")

            # Save Camera_Container offset value
            camera_offset = bpy.context.scene.objects["Camera_Container"].constraints["Follow Path"].offset
            camera_coords_filepath = output_dir_camera_coords / f"{split_name}_{str(current_start_idx).zfill(6)}.txt"
            with open(camera_coords_filepath, "w") as f:
                f.write(f"{camera_offset:.6f}\n")

            # Save debris orientation in Euler angles
            euler_angles = q.to_euler()
            debris_orientation_euler = (euler_angles.x, euler_angles.y, euler_angles.z)
            debris_orientation_euler_filepath = output_dir_debris_orientation_euler / f"{split_name}_{str(current_start_idx).zfill(6)}.txt"
            with open(debris_orientation_euler_filepath, "w") as f:
                f.write(f"{debris_orientation_euler[0]:.6f} {debris_orientation_euler[1]:.6f} {debris_orientation_euler[2]:.6f}\n")

            # Save debris orientation in quaternion format
            quaternion_angles = q
            debris_orientation_quaternion = (quaternion_angles.w, quaternion_angles.x, quaternion_angles.y, quaternion_angles.z)
            debris_orientation_quaternion_filepath = output_dir_debris_orientation_quaternion / f"{split_name}_{str(current_start_idx).zfill(6)}.txt"
            with open(debris_orientation_quaternion_filepath, "w") as f:
                f.write(f"{debris_orientation_quaternion[0]:.6f} {debris_orientation_quaternion[1]:.6f} {debris_orientation_quaternion[2]:.6f} {debris_orientation_quaternion[3]:.6f}\n")

            # Hide the rendered debris object
            if debris_name != 'Empty':
                debris_to_render.hide_render = True

            # Update indices
            current_start_idx += 1
            start_index_local += 1

            # Check if the current batch is complete
            if start_index_local >= BATCH_LIMIT:
                state["batch"] += 1
                state["start_idx"] = current_start_idx
                state["no_of_batches"] += 1
                save_state_and_exit(state, state_file)
                return

    # Re-enable rendering for all debris objects
    for name in debris_names:
        if name != 'Empty':
            bpy.context.scene.objects[name].hide_render = False

    # Report total rendering time
    stop_time = time.time()
    total_time = stop_time - start_time
    print(f"Total Rendering Time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")

#------------------------------------------------------------------------------
# Script Entry Point
#------------------------------------------------------------------------------
def main():
    """
    Main entry point for the script.

    This function loads the current state, sets up the scene by retrieving key objects,
    assigns class IDs to debris objects, and initiates the rendering process.
    """
    # Load existing state or initialize defaults
    state, state_file = load_or_initialize_state()

    # Retrieve axis objects for random rotation
    axis1_obj = check_object_in_scene('Earth_Axis')
    axis2_obj = check_object_in_scene('Orbit_Container_Axis')
    axes = [axis1_obj, axis2_obj]

    # Retrieve additional objects for rotation adjustments
    clouds_obj = check_object_in_scene('Clouds')
    sun_obj = check_object_in_scene('Sun')

    # Define debris objects and assign class IDs
    debris_names = DEBRIS_NAMES
    assign_classes_to_debris(debris_names)

    # Begin the rendering process with configured image count
    render_dataset(state, state_file, debris_names, TOTAL_IMAGES_PER_DEBRIS, axes, clouds_obj, sun_obj)

#------------------------------------------------------------------------------
# Execute main function when script is run
#------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
