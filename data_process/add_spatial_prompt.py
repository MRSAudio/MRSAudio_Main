import numpy as np
from scipy.signal import savgol_filter, medfilt
from openai import OpenAI
from typing import List, Dict, Tuple
import json
import numpy as np
from scipy.signal import savgol_filter, medfilt
from openai import OpenAI
from typing import List, Dict, Tuple
import json
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os


def analyze_motion(coords: np.ndarray, 
                  velocity_threshold: float = 0.1,
                  dynamic_window: int = 5) -> Tuple[List[str], List[Tuple[int, int]]]:
    velocities = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    motion_states = []
    
    for i in range(len(velocities)):
        window_start = max(0, i - dynamic_window)
        recent_vels = velocities[window_start:i+1]
        if velocities[i] > velocity_threshold or np.std(recent_vels) > 0.02:
            motion_states.append("moving")
        else:
            motion_states.append("static")

    direction_changes = []
    segment_start = 0
    
    segments = []
    for i in range(1, len(motion_states)):
        if motion_states[i] != motion_states[i-1]:
            segments.append((motion_states[i-1], segment_start, i))
            segment_start = i
    segments.append((motion_states[-1], segment_start, len(motion_states)))
    
    last_direction = None
    for state, start, end in segments:
        if state == "moving" and (end - start) >= 3:
            delta = coords[end] - coords[start]
            current_direction = np.sign(delta).astype(int)
            
            if last_direction is not None and not np.array_equal(current_direction, last_direction):
                direction_changes.append((start, end))
            last_direction = current_direction
            
    return motion_states, direction_changes

def call_openai_client(api_key: str, prompt: str) -> str:
    client = OpenAI(
        api_key=api_key,
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a trajectory analysis assistant. Output English descriptions."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=300,
        stream=False
    )
    
    return response.choices[0].message.content

def generate_motion_prompt(
    raw_coords: np.ndarray,
    api_key: str
) -> str:
    coords = raw_coords
    motion_states, baf_points = analyze_motion(coords)
    timestamps = [f"{i*0.05:.2f}s: {tuple(coord)}" for i, coord in enumerate(coords)]
    
    prompt_template = """
        Given the refined spatial audio trajectory data:
        Head Position: (0, 0, 0), Orientation: The direction of the head is towards the positive direction of the y-axis, and the top of the head is defined as the z-axis.
        Smoothed Coordinates: {coords}.
        Motion Analysis: 
        - Static/Dynamic States: {motion_states}
        - Back-and-Forth Points: {baf_points}
        - Coordinate Mapping Rules: 
            1. X-axis: Right(+) / Left(-)
            2. Y-axis: Front(+) / Back(-)
            3. Z-axis: Up(+) / Down(-)

        Please analyze and generate a concise English description following these rules:
        1. Determine source status: [Static] if all motion_states are 'static', else [Dynamic]
        2. Describe initial position relative to listener using combined direction labels (e.g., "front-left", "back-right up", "center down")
        3. For motion path: 
        - Start with "Moves from [direction]" 
        - Add key transition points for complex paths (e.g., "via center back" or "passing through right front level")
        - End with final position (e.g., "to full left side")
        4. Time conversion: Convert frame indices in baf_points to seconds (frame/20) with ±0.1s tolerance
        5. Motion path may consist of several segments. (e.g., "move from front to further front and then mve from front to front-left")

        Critical Constraints:
        - Ignore micro-jitters (<0.1m movements)
        - Use only combined directional terms (no raw coordinates)
        - Round time values to 1 decimal place
        - Keep total output under 30 words
        
        Output format example: 
        "[STATIC] Source locates at [initial direction] and pauses in [initial direction] quadrant"
        "[DYNAMIC] Travel from back-left to front-center, maintaining stable vertical position and then pause"
            
        """
    prompt = prompt_template.format(
        coords="\n".join(timestamps[:10] + ["..."] + timestamps[-10:]),
        motion_states=" ".join([f"{state}@{i}" for i, state in enumerate(motion_states)]),
        baf_points=", ".join([f"{int(start/20)}s-{int(end/20)}s" for start, end in baf_points])
    )
    
    return call_openai_client(api_key, prompt)


json_file = "your meta.json"
target_json_file = "your target.json"


def process_single_item(args):
    key, value, api_key = args
    try:
        npy_path = value['pos_fn']
        if not os.path.exists(npy_path):
            return key, {"error": f"npy file does not exist: {npy_path}"}
        
        sample_coords = np.load(npy_path, allow_pickle=True)[:, :3]
        if False:
            result = get_static_prompt(
                raw_coords=sample_coords,
                api_key=api_key
            )
            return key, {"spatial_prompt": result}
        else:
            result = generate_motion_prompt(
                raw_coords=sample_coords,
                api_key=api_key
            )
            return key, {"spatial_prompt": result}
    
    except Exception as e:
        return key, {"error": str(e)}

def main_parallel():
    api_key = "your_api_key" 
    workers = 44

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    task_args = [(k, v, api_key) for k, v in data.items()]

    print(len(task_args))
    with Pool(processes=workers) as pool:
        results = []
        with tqdm(total=len(task_args), desc="generate spatial prompt") as pbar:
            for key, result in pool.imap_unordered(process_single_item, task_args):
                data[key].update(result)
                pbar.update(1)
                if "error" in result:
                    pbar.write(f"error {key}: {result['error']}")

    with open(target_json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Results save in {target_json_file}")

if __name__ == "__main__":
    main_parallel()