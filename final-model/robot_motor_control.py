"""
Clean Motor Controller for AUPPBot
Maps emotions to robot actions with smooth control
"""

import time
from enum import Enum

try:
    from auppbot import AUPPBot
    AUPPBOT_AVAILABLE = True
except ImportError:
    AUPPBOT_AVAILABLE = False
    print("⚠ AUPPBot not available - simulation mode only")


class RobotAction(Enum):
    """Robot action types."""
    FORWARD = "forward"
    BACKWARD = "backward"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    STOP = "stop"


class EmotionRobotController:
    """
    Robot controller that maps emotions to motor actions.
    
    Emotion → Action Mapping:
    - Happy → Move Forward
    - Angry → Move Backward
    - Surprised → Turn Left
    - Sad → Turn Right
    - Neutral → Stop
    
    Modes:
    - reactive: Continuous control (action changes with emotion)
    - timed: Execute action for fixed duration, then wait
    """
    
    def __init__(self, simulation=True, port="/dev/ttyUSB0", baudrate=115200, 
                 base_speed=25, turn_duration=0.1, use_servos=True, 
                 servo1_center=90, servo2_center=90, 
                 control_mode='timed', action_duration=6.0):
        """
        Initialize robot controller.
        
        Args:
            simulation: If True, only print actions (no hardware)
            port: Serial port for AUPPBot
            baudrate: Serial baudrate
            base_speed: Base motor speed (0-100)
            turn_duration: Duration for each turn command (seconds)
            use_servos: Whether to control servos (e.g., for camera gimbal)
            servo1_center: Center position for servo1 (0-180)
            servo2_center: Center position for servo2 (0-180)
            control_mode: 'reactive' (continuous) or 'timed' (duration-based)
            action_duration: How long to execute each action in timed mode (seconds)
        """
        self.simulation = simulation
        self.base_speed = base_speed
        self.turn_duration = turn_duration
        self.current_action = RobotAction.STOP
        self.bot = None
        self.use_servos = use_servos
        self.servo1_center = servo1_center
        self.servo2_center = servo2_center
        self.control_mode = control_mode
        self.action_duration = action_duration
        
        # Timed mode state
        self.is_executing = False
        self.action_start_time = 0
        self.last_emotion = None
        
        # Emotion mapping (by class name)
        self.emotion_to_action = {
            'angry': RobotAction.BACKWARD,
            'happy': RobotAction.FORWARD,
            'neutral': RobotAction.STOP,
            'sad': RobotAction.TURN_RIGHT,
            'surprised': RobotAction.TURN_LEFT,
        }
        
        # Initialize hardware if not simulation
        if not simulation and AUPPBOT_AVAILABLE:
            try:
                self.bot = AUPPBot(port, baudrate, auto_safe=True)
                
                # Center servos on startup if enabled
                if self.use_servos:
                    self.bot.servo1.angle(self.servo1_center)
                    self.bot.servo2.angle(self.servo2_center)
                
                print(f"✓ AUPPBot connected on {port}")
            except Exception as e:
                print(f"❌ Failed to connect to AUPPBot: {e}")
                print("   Falling back to simulation mode")
                self.simulation = True
        else:
            print("⚙ Motor Controller initialized in SIMULATION mode")
    
    def execute_emotion(self, emotion: str):
        """
        Execute action based on emotion.
        
        Args:
            emotion: Emotion name ('angry', 'happy', 'neutral', 'sad', 'surprised')
        """
        if self.control_mode == 'reactive':
            self._execute_reactive(emotion)
        else:  # timed mode
            self._execute_timed(emotion)
    
    def _execute_reactive(self, emotion: str):
        """Reactive mode: action changes immediately with emotion."""
        action = self.emotion_to_action.get(emotion, RobotAction.STOP)
        
        if action != self.current_action:
            self.current_action = action
            self._execute_action(action)
    
    def _execute_timed(self, emotion: str):
        """
        Timed mode: execute action for fixed duration, then wait for new emotion.
        
        Behavior:
        - If not executing and new emotion detected: start action for duration
        - If executing: ignore new emotions until duration completes
        - After duration: stop and wait for different emotion
        """
        current_time = time.time()
        
        # Check if currently executing an action
        if self.is_executing:
            elapsed = current_time - self.action_start_time
            
            if elapsed >= self.action_duration:
                # Action duration complete - stop and reset
                self.stop()
                self.is_executing = False
                self.last_emotion = None
                
                if self.simulation:
                    print(f"  ⏱ Action complete ({self.action_duration}s)")
            
            # Still executing - ignore new emotions
            return
        
        # Not executing - check for new emotion
        if emotion != self.last_emotion and emotion is not None:
            # New emotion detected - start action
            action = self.emotion_to_action.get(emotion, RobotAction.STOP)
            
            if action != RobotAction.STOP:  # Don't execute timed "stop"
                self.current_action = action
                self.is_executing = True
                self.action_start_time = current_time
                self.last_emotion = emotion
                
                if self.simulation:
                    print(f"\n🎯 New emotion: {emotion.upper()} - executing for {self.action_duration}s")
                
                self._execute_action(action)
            else:
                # Neutral = immediate stop
                self.stop()
                self.last_emotion = emotion
    
    def _execute_action(self, action: RobotAction):
        """Execute the motor action."""
        if self.simulation:
            self._print_action(action)
        else:
            self._hardware_action(action)
    
    def _print_action(self, action: RobotAction):
        """Print action for simulation."""
        symbols = {
            RobotAction.FORWARD: "↑ FORWARD",
            RobotAction.BACKWARD: "↓ BACKWARD",
            RobotAction.TURN_LEFT: "← TURN LEFT",
            RobotAction.TURN_RIGHT: "→ TURN RIGHT",
            RobotAction.STOP: "■ STOP",
        }
        print(f"🤖 Robot: {symbols[action]}")
    
    def _hardware_action(self, action: RobotAction):
        """Execute action on AUPPBot hardware."""
        if not self.bot:
            return
        
        speed = self.base_speed
        
        # Keep servos centered (or adjust based on face position if tracking)
        if self.use_servos:
            self.bot.servo1.angle(self.servo1_center)
            self.bot.servo2.angle(self.servo2_center)
        
        if action == RobotAction.FORWARD:
            # All wheels forward
            self.bot.motor1.speed(speed)
            self.bot.motor2.speed(speed)
            self.bot.motor3.speed(speed)
            self.bot.motor4.speed(speed)
            
        elif action == RobotAction.BACKWARD:
            # All wheels backward
            self.bot.motor1.speed(-speed)
            self.bot.motor2.speed(-speed)
            self.bot.motor3.speed(-speed)
            self.bot.motor4.speed(-speed)
            
        elif action == RobotAction.TURN_LEFT:
            # Left wheels backward, right wheels forward
            self.bot.motor1.speed(-speed)
            self.bot.motor2.speed(-speed)
            self.bot.motor3.speed(speed)
            self.bot.motor4.speed(speed)
            
        elif action == RobotAction.TURN_RIGHT:
            # Left wheels forward, right wheels backward
            self.bot.motor1.speed(speed)
            self.bot.motor2.speed(speed)
            self.bot.motor3.speed(-speed)
            self.bot.motor4.speed(-speed)
            
        elif action == RobotAction.STOP:
            self.stop()
    
    def stop(self):
        """Stop all motors."""
        self.current_action = RobotAction.STOP
        
        if self.simulation:
            print("🤖 Robot: ■ STOP")
        elif self.bot:
            self.bot.stop_all()
    
    def get_current_action(self):
        """Get current action as string for UI display."""
        action_names = {
            RobotAction.FORWARD: "Forward",
            RobotAction.BACKWARD: "Backward",
            RobotAction.TURN_LEFT: "Left",
            RobotAction.TURN_RIGHT: "Right",
            RobotAction.STOP: "Stop"
        }
        return action_names.get(self.current_action, "Unknown")
    
    def cleanup(self):
        """Cleanup and stop robot."""
        self.stop()
        if self.bot:
            self.bot.stop_all()


# Test the controller
if __name__ == '__main__':
    print("=" * 60)
    print("EMOTION ROBOT CONTROLLER TEST")
    print("=" * 60)
    
    controller = EmotionRobotController(simulation=True)
    
    # Test emotion sequence
    test_emotions = [
        'happy', 'happy', 'happy',      # Forward
        'sad', 'sad',                    # Turn right
        'happy', 'happy',                # Forward
        'surprised', 'surprised',        # Turn left
        'neutral',                       # Stop
        'angry', 'angry',                # Backward
        'neutral',                       # Stop
    ]
    
    print("\nTesting emotion sequence:")
    for emotion in test_emotions:
        print(f"\nEmotion: {emotion.upper()}")
        controller.execute_emotion(emotion)
        time.sleep(0.5)
    
    controller.stop()
    print("\n✓ Test complete!")
    print("=" * 60)
