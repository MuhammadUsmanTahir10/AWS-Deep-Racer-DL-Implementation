# AWS-Deep-Racer-DL-Models

Implementing different reward functions for optimized model generation - AWS Deep Racer League 

**What is Deep Racer**

DeepRacer, powered by AWS, redefines the concept of racing by offering a platform for training and testing self-driving model cars in a virtual environment. This cutting-edge methodology employs reinforcement learning, where favorable behaviors, such as staying on course, are rewarded, while undesirable actions face penalties. With DeepRacer, you're in the driver's seat when it comes to defining what qualifies as good and bad behavior, thanks to a Python-based reward function. What's more, you can craft and customize a range of training scenarios, choosing from various virtual tracks and adjusting training duration's to suit your preferences. You even have granular control over your car's steering and acceleration settings. Throughout the training process, you'll closely monitor progress by tracking the percentage of the track completed before any off-road incidents occur. Following each training session, you'll be able to comprehensively assess your car's performance by reviewing virtual evaluation movies, offering invaluable insights into its behavior.

**Lets Begin**

My approach to training the DeepRacer model began with simplicity, using basic reward functions to establish a foundation. As the training progressed, I meticulously fine-tuned the model, continuously monitoring its development. After experimenting with several custom reward functions, I opted to start with one of AWS's baseline functions, "follow the tangent," as a starting point. This decision allowed us to create a model that could consistently complete the track, serving as a solid base for further enhancements. Throughout the initial phases of our work, we retrained the models to gain a clear understanding of how each component of the reward function impacted overall performance. An intriguing discovery emerged; we found that adhering to the center-line had minimal impact when utilized with the base model. This revelation prompted us to explore various approaches to reward functions and devise strategies aimed at optimizing the model's performance. For a comprehensive overview of our diverse reward functions and detailed strategies, please refer to the sections below.

**Follow the Center Line**

In the realm of DeepRacer, the reward function serves as the guiding force shaping the behavior of autonomous model cars. This specific reward function, showcased as an example, places a strong emphasis on incentivizing the agent to follow the center line of the track. As we delve into the code, we will analyze its inner workings, detailing how it influences the car's actions during training, and reflect upon the outcomes it produces.

Lets begin with this baseline code for following the center line.

def reward_function(params):
    '''
    Example of rewarding the agent to follow the center line
    '''
    
    # Read input parameters
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']

    # Calculate 3 markers that are increasingly further away from the center line
    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    marker_3 = 0.5 * track_width

    # Give a higher reward if the car is closer to the center line and vice versa
    if distance_from_center <= marker_1:
        reward = 1
    elif distance_from_center <= marker_2:
        reward = 0.5
    elif distance_from_center <= marker_3:
        reward = 0.1
    else:
        reward = 1e-3  # likely crashed/close to off-track

    return reward

Now, let's break down the components of this reward function:

params is a dictionary containing various input parameters that provide information about the car's state.
track_width provides the width of the track.
distance_from_center indicates how far the car is from the center of the track.

The implementation of this reward function has yielded promising results in training DeepRacer models. By motivating the agent to follow the center line, it has consistently produced cars with an excellent track-following capability. During training, the models demonstrated an impressive ability to stay close to the desired path, maintaining a strong track position. These outcomes underscore the significance of a well-tailored reward function in shaping DeepRacer behavior and showcase the potential for fine-tuning and further optimizations to achieve even more exceptional performance in subsequent iterations.

**Stay Inside the Two Borders**

In the realm of DeepRacer, crafting a well-structured reward function is fundamental to training autonomous model cars. This specific reward function, showcased as an example, focuses on encouraging the agent to remain within the two borders of the track. As we delve into the code, we will dissect its components and logic to understand how it influences the car's behavior during training and reflect upon the outcomes it generates.

Lets delve into code:

def reward_function(params):
    '''
    Example of rewarding the agent to stay inside the two borders of the track
    '''
    
    # Read input parameters
    all_wheels_on_track = params['all_wheels_on_track']
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    
    # Give a very low reward by default
    reward = 1e-3

    # Give a high reward if no wheels go off the track and 
    # the car is somewhere in between the track borders 
    if all_wheels_on_track and (0.5*track_width - distance_from_center) >= 0.05:
        reward = 1.0

    # Always return a float value
    return reward

Now, let's break down the components of this reward function:

params is a dictionary containing various input parameters that provide information about the car's state.
all_wheels_on_track indicates whether all wheels of the car are on the track.
distance_from_center tells us how far the car is from the center of the track.
track_width provides the width of the track.

The reward function begins by setting a low default reward (1e-3). It then evaluates two conditions: whether all wheels are on the track and whether the car is sufficiently close to the center of the track (within 5% of the track's width). If both conditions are met, the function assigns a high reward of 1.0. Finally, it ensures that a float value is always returned as the reward.

The implementation of this reward function has produced noteworthy results in training DeepRacer models. During training, the autonomous car consistently demonstrated a remarkable ability to stay within the track boundaries. By incentivizing the agent to remain on the track and penalizing deviations, the function effectively guided the car's behavior. These outcomes underscore the significance of a well-constructed reward function in shaping the behavior of DeepRacer models and highlight the potential for further refinements and optimizations to achieve even more remarkable results in future iterations.

**Prevent Zig-Zag**

In the realm of DeepRacer, the reward function plays a pivotal role in shaping the behavior of autonomous model cars. This particular reward function serves as an example of penalizing excessive steering, a strategy employed to mitigate zig-zag behaviors often observed during training. As we delve into the code, we will dissect its components and logic, uncovering how it influences the car's actions throughout the training process and reflecting upon the outcomes it generates.

Lets look at code:

def reward_function(params):
    '''
    Example of penalize steering, which helps mitigate zig-zag behaviors
    '''
    
    # Read input parameters
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    abs_steering = abs(params['steering_angle']) # Only need the absolute steering angle

    # Calculate 3 marks that are farther and farther away from the center line
    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    marker_3 = 0.5 * track_width

    # Give a higher reward if the car is closer to the center line and vice versa
    if distance_from_center <= marker_1:
        reward = 1.0
    elif distance_from_center <= marker_2:
        reward = 0.5
    elif distance_from_center <= marker_3:
        reward = 0.1
    else:
        reward = 1e-3  # likely crashed/close to off-track

    # Steering penalty threshold, change the number based on your action space setting
    ABS_STEERING_THRESHOLD = 15 

    # Penalize reward if the car is steering too much
    if abs_steering > ABS_STEERING_THRESHOLD:
        reward *= 0.8

    return float(reward)

Now, let's break down the components of this reward function:

params is a dictionary containing various input parameters that provide information about the car's state.
distance_from_center indicates how far the car is from the center of the track.
track_width provides the width of the track.
abs_steering calculates the absolute value of the car's steering angle.

The reward function calculates three markers, each progressively farther from the center line. These markers serve as reference points for the car's position relative to the center line. The function assigns rewards based on the car's distance from these markers. If the car is very close to the center line (within marker_1), it receives a high reward (1.0), indicating optimal performance. As the car deviates further from the center line, the reward diminishes, with a value of 0.1 assigned if it's close to the track's edge. If the car strays too far from the center (beyond marker_3), it receives a low default reward (1e-3), indicating a potential crash or near-off-track situation. Additionally, the function penalizes the reward if the car's absolute steering angle exceeds a predefined threshold (ABS_STEERING_THRESHOLD). Excessive steering is discouraged, as it can lead to erratic zig-zag behaviors during training.

By penalizing excessive steering and encouraging the agent to stay close to the center line, it has effectively mitigated zig-zag behaviors. During training, the models demonstrated improved stability and a more controlled path-following behavior. These outcomes highlight the importance of a well-crafted reward function in influencing DeepRacer models and offer potential avenues for further refinement and optimization to achieve even more exceptional performance in future iterations.

**Stay in One Lane without Crashing into Obstacles**

This reward function is a complex example, designed to encourage the agent to remain within two track borders while penalizing closeness to objects in its path. As we delve into the code, we will dissect its components and logic, uncovering how it influences the car's actions during training and reflect upon the outcomes it generates.

Lets discuss code:

import math

def reward_function(params):
    '''
    Example of rewarding the agent to stay inside two borders
    and penalizing getting too close to the objects in front
    '''
    all_wheels_on_track = params['all_wheels_on_track']
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    objects_location = params['objects_location']
    agent_x = params['x']
    agent_y = params['y']
    _, next_object_index = params['closest_objects']
    objects_left_of_center = params['objects_left_of_center']
    is_left_of_center = params['is_left_of_center']
    
    # Initialize reward with a small number but not zero
    # because zero means off-track or crashed
    reward = 1e-3
    
    # Reward if the agent stays inside the two borders of the track
    if all_wheels_on_track and (0.5 * track_width - distance_from_center) >= 0.05:
        reward_lane = 1.0
    else:
        reward_lane = 1e-3
    
    # Penalize if the agent is too close to the next object
    reward_avoid = 1.0
    
    # Distance to the next object
    next_object_loc = objects_location[next_object_index]
    distance_closest_object = math.sqrt((agent_x - next_object_loc[0])**2 + (agent_y - next_object_loc[1])**2)
    
    # Decide if the agent and the next object are on the same lane
    is_same_lane = objects_left_of_center[next_object_index] == is_left_of_center
    
    if is_same_lane:
        if 0.5 <= distance_closest_object < 0.8:
            reward_avoid *= 0.5
        elif 0.3 <= distance_closest_object < 0.5:
            reward_avoid *= 0.2
        elif distance_closest_object < 0.3:
            reward_avoid = 1e-3  # Likely crashed
    
    # Calculate reward by putting different weights on
    # the two aspects above
    reward += 1.0 * reward_lane + 4.0 * reward_avoid
    return reward

Now, let's break down the components of this reward function:

params is a dictionary containing various input parameters that provide information about the car's state.
all_wheels_on_track indicates whether all wheels of the car are on the track.
distance_from_center tells us how far the car is from the center of the track.
track_width provides the width of the track.
objects_location contains the locations of objects in front of the car.
agent_x and agent_y represent the coordinates of the agent's position.
closest_objects contains information about the closest objects.
objects_left_of_center indicates whether objects are left of the center.
is_left_of_center represents whether the agent is left of the center.

The reward function starts by initializing a base reward (1e-3), which is greater than zero to signify that the agent is still on the track. It then evaluates two key aspects:

Reward for Staying in Lane (reward_lane): This aspect rewards the agent for staying within two track borders. If all wheels are on the track and the agent is within a certain distance from the center line (0.5 * track_width - distance_from_center >= 0.05), it receives a high reward (1.0). Otherwise, it gets a low default reward (1e-3), indicating a potential off-track or crashed situation.

Penalty for Avoiding Objects (reward_avoid): This aspect penalizes the agent for getting too close to the next object in its path. The distance to the next object is calculated, and the penalty varies based on this distance and whether the agent and the object are in the same lane. The closer the object, the more significant the penalty, with the penalty becoming severe (1e-3) if the agent is too close, indicating a likely crash.

The final reward is calculated by combining both aspects, with different weights assigned. The agent is incentivized to stay in the lane and avoid objects, striking a balance between track-following behavior and collision avoidance.

The implementation of this reward function has produced notable results during training. By simultaneously rewarding the agent for staying within the track boundaries and penalizing close encounters with objects, it has achieved a delicate balance between track-following and obstacle avoidance. This has resulted in models that exhibit impressive.

**Formulating Complex Reward Functions**

My journey into intricate reward functions commenced with a pivotal decision—to diversify my approach. After experimenting with numerous custom reward functions, I made a strategic choice to initiate my exploration by embracing one of AWS's foundational functions, aptly named "follow the tangent." This choice marked the inception of my quest to elevate my model's performance. Initially, my primary goal was to construct a model with the consistent capability to conquer the track. Once I solidified this fundamental groundwork, my mission evolved into a meticulous process of refinement and expansion upon the initial function. My aim was to find a delicate equilibrium among various reward components.

As my journey unfolded, I grew acutely aware of the imperative nature of comprehensively assessing the impact of each reward function component on overall performance. To attain this, I dedicated considerable effort to retraining my models during the initial phase of my work, diligently unraveling the intricate influences of these components on my agent's behavior. One remarkable revelation emerged along this path—closely adhering to the centerline strategy bore minimal fruit when paired with the base model, underscoring the imperative for a more nuanced approach. Ultimately, my expedition culminated in the creation of a comprehensive reward function that harmoniously integrated various elements, including a speed reward. This achievement represents a well-rounded approach that now expertly guides my DeepRacer model's behavior.

In my quest to maximize the potential of my DeepRacer model, I embarked on a mission marked by intricate decision-making. My approach was tailored for ease of tuning and evaluation, ensuring that each reward component could be fine-tuned intuitively. This journey began with the adoption of a fundamental AWS baseline function known as "follow the tangent," setting the stage for my pursuit of model performance enhancement. The initial phase of my efforts was dedicated to building a model capable of consistently conquering the track. Once this foundational achievement was in place, my focus shifted towards refining and expanding upon the initial reward function. The goal was clear—to strike a harmonious balance among the various reward components.

As I progressed, I recognized the significance of thoroughly gauging the impact of each reward function component on the overall performance. This prompted a deliberate and extensive phase of model retraining, designed to unravel the intricate influences of these components on the behavior of my agent. Amidst this journey, a pivotal revelation emerged—the strategy of strictly adhering to the centerline had limited effectiveness when combined with the base model. This revelation underscored the necessity for a more nuanced approach. The culmination of my efforts resulted in the development of a comprehensive reward function that integrated various elements, including a speed reward component. This achievement represented a well-rounded approach that expertly guided my DeepRacer model's behavior.

In tandem with enhancing performance, I ventured into adapting my model for the unpredictabilities of a "real-life" environment. This entailed retraining the same model on different tracks, with a preference for those closely resembling the original one, such as "Cumulo" and "Empire." The final step in this phase was an extensive 5-hour retraining session on the re:Invent 2018 track. During actual races, I confronted the challenge of adjusting the car's speed while in motion, introducing an element of unpredictability not encountered during training. My solution was to implement gradual speed adjustments, lap by lap, starting with a slower speed (60%) and gradually increasing it to 85%. Remarkably, the model exhibited the ability to autonomously adapt its speed according to the track's contours once left to its own devices.

This journey of discovery and innovation underscored the adaptability and potential of DeepRacer models, setting the stage for further exploration and optimization in the world of autonomous racing

Lets look at reward function for this model:

def reward_function(params):

    # Reward weights
    speed_weight = 100
    heading_weight = 100
    steering_weight = 50

    # Initialize the reward based on current speed
    max_speed_reward = 10 * 10
    min_speed_reward = 3.33 * 3.33
    abs_speed_reward = params['speed'] * params['speed']
    speed_reward = (abs_speed_reward - min_speed_reward) / (max_speed_reward - min_speed_reward) * speed_weight
    
    # - - - - - 
    
    # Penalize if slow speed action space
    if not params['speed'] < 5:
     	return 1e-3    

    # Penalize if the car goes off track
    if not params['all_wheels_on_track']:
        return 1e-3
    
    # - - - - - 
    
    # Calculate the direction of the center line based on the closest waypoints
    next_point = params['waypoints'][params['closest_waypoints'][1]]
    prev_point = params['waypoints'][params['closest_waypoints'][0]]

    # Calculate the direction in radius, arctan2(dy, dx), the result is (-pi, pi) in radians
    track_direction = math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0]) 
    # Convert to degree
    track_direction = math.degrees(track_direction)

    # Calculate the difference between the track direction and the heading direction of the car
    direction_diff = abs(track_direction - params['heading'])
    if direction_diff > 180:
        direction_diff = 360 - direction_diff
    
    abs_heading_reward = 1 - (direction_diff / 180.0)
    heading_reward = abs_heading_reward * heading_weight
    
    # - - - - -
    
    # Reward if steering angle is aligned with direction difference
    abs_steering_reward = 1 - (abs(params['steering_angle'] - direction_diff) / 180.0)
    steering_reward = abs_steering_reward * steering_weight

    # - - - - -
    
    return speed_reward + heading_reward + steering_reward

Now lets discuss the code:

Input Parameters: The function takes a dictionary called params as its input. This dictionary contains information about the agent's state and the track.
all_wheels_on_track: Indicates whether all wheels of the car are on the track.
distance_from_center: Tells us how far the car is from the center of the track.
track_width: Provides the width of the track.
objects_location: Contains the locations of objects in front of the car.
agent_x and agent_y: Represent the coordinates of the agent's position.
closest_objects: Contains information about the closest objects.
objects_left_of_center: Indicates whether objects are left of the center.
is_left_of_center: Represents whether the agent is left of the center.
Reward Initialization: The function initializes the reward with a small number (1e-3), but not zero, to distinguish between being off-track or crashed and being on the track.
Reward for Staying in Lane (reward_lane): The function calculates a reward for the agent staying within the two borders of the track. If all wheels are on the track and the agent is within a certain distance from the center line (0.5 * track_width - distance_from_center >= 0.05), it receives a high reward (1.0). Otherwise, it gets a low default reward (1e-3).
Penalty for Avoiding Objects (reward_avoid): The function penalizes the agent for getting too close to the next object in its path. The penalty varies based on the distance to the next object and whether the agent and the object are in the same lane.
Distance Calculation: The function calculates the distance to the next object using the agent's and object's coordinates.
Same Lane Check: It checks if the agent and the next object are in the same lane (i.e., both on the left or right of the center).
Distance-Based Penalties: Depending on the distance to the next object and the same lane condition, the function adjusts the reward_avoid value, introducing penalties for getting too close.
Final Reward Calculation: The final reward is calculated by combining both the reward_lane and reward_avoid components with different weights

This reward function encourages the agent to stay within the track boundaries and penalizes it for getting too close to objects. The specific distances and penalties are fine-tuned to achieve the desired behavior. The result is a reward function that guides the DeepRacer agent to navigate the track while avoiding collisions with objects in its path, striking a balance between speed and safety. The exact performance of this function depends on the specific track and agent configuration but aims to optimize completion rates and lap times.


