# config.py
# Pattern Detector V8.2.1 - Configuration Settings
# SWING TRADING OPTIMIZED - 1-3 Week Hold Periods

# Application Settings
APP_TITLE = "Pro Pattern Detector v8.2.1"
APP_SUBTITLE = "Swing Trading Optimized - Professional Pattern Recognition"
VERSION = "8.2.1"

# Pattern Detection Settings
PATTERNS = ["Flat Top Breakout", "Bull Flag", "Cup Handle", "Inside Bar", "Inverse Head Shoulders"]
DEFAULT_PATTERNS = ["Flat Top Breakout", "Bull Flag", "Inside Bar", "Inverse Head Shoulders"]

# Timeframe Settings
PERIOD_OPTIONS = ["1mo", "3mo", "6mo", "1y", "1wk (Weekly)"]
DEFAULT_PERIOD = "3mo"

# Confidence Settings
MIN_CONFIDENCE_RANGE = (45, 85)
DEFAULT_MIN_CONFIDENCE = 55

# Volume Analysis Settings
VOLUME_THRESHOLDS = {
    "good": 1.3,
    "strong": 1.5,
    "exceptional": 2.0
}

VOLUME_SCORE_POINTS = {
    "exceptional": 25,  # 2.0x+ average
    "strong": 20,       # 1.5-2.0x average
    "good": 15,         # 1.3-1.5x average
    "weak": 0           # <1.3x average
}

# Pattern-specific volume bonuses
PATTERN_VOLUME_BONUS = {
    "Bull Flag": 20,      # Flagpole vs flag volume
    "Cup Handle": 20,     # Volume dryup in handle
    "Flat Top Breakout": 20,  # Breakout volume surge
    "Inside Bar": 15,     # Consolidation volume
    "Inverse Head Shoulders": 20  # Classic volume diminishing pattern
}

# Confidence Capping
MAX_CONFIDENCE_WITHOUT_VOLUME = 70

# Market Timing Settings
MARKET_TIMING_ADJUSTMENTS = {
    "weekend_penalty": 0.95,      # -5%
    "friday_penalty": 0.85,       # -15% without exceptional volume
    "midweek_bonus": 1.02,        # +2%
    "monday_gap_check": True
}

# Pattern Age Limits (days/weeks) - OPTIMIZED FOR SWING TRADING
PATTERN_AGE_LIMITS = {
    "daily": {
        "Flat Top Breakout": 8,
        "Bull Flag": 10,
        "Cup Handle": 35,              # CHANGED from 30 - allows fuller base formation
        "Inside Bar": 6,
        "Inverse Head Shoulders": 50   # CHANGED from 30 - H&S takes longer to form
    },
    "weekly": {
        "Flat Top Breakout": 8,  # weeks
        "Bull Flag": 10,         # weeks
        "Cup Handle": 25,        # CHANGED from 30 - tighter for weekly
        "Inside Bar": 8,         # weeks
        "Inverse Head Shoulders": 35  # CHANGED from 20 - allows proper formation
    }
}

# Risk Management Settings
RISK_MULTIPLIERS = {
    "volatility_stop": 1.5,  # 1.5x average daily range
    "min_stop_distance": {
        "Flat Top Breakout": 0.03,  # 3%
        "Bull Flag": 0.04,          # 4%
        "Cup Handle": 0.05,         # 5%
        "Inside Bar": 0.05,         # 5%
        "Inverse Head Shoulders": 0.04  # 4%
    }
}

# Target Calculation Settings
MIN_RISK_REWARD_RATIOS = {
    "target1": 1.5,
    "target2": 2.5
}

# Inside Bar specific settings
INSIDE_BAR_CONFIG = {
    "entry_buffer": 1.05,    # 5% above inside bar high
    "stop_buffer": 0.95,     # 5% below inside bar low
    "target2_multiplier": 1.13,  # 13% above mother bar high
    "target3_multiplier": 1.21,  # 21% above mother bar high
    "max_inside_bars": 2,
    "preferred_inside_bars": 1
}

# Technical Indicator Settings
INDICATOR_PERIODS = {
    "rsi": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "sma": 20,
    "volume_sma": 20
}

# Pattern Detection Thresholds - SWING TRADING OPTIMIZED
PATTERN_THRESHOLDS = {
    "Flat Top Breakout": {
        "min_initial_gain": 0.10,  # 10%
        "min_pullback": 0.08,      # 8%
        "resistance_tolerance": 0.98  # 98% of resistance level
    },
    "Bull Flag": {
        "min_flagpole_gain": 0.08,  # 8%
        "pullback_range": (-0.15, 0.05),  # -15% to +5%
        "flag_tolerance": 0.95
    },
    "Cup Handle": {
        "min_cup_depth": 0.08,      # 8% - catches early bases
        "max_cup_depth": 0.30,      # 30% - CHANGED from 60% (swing trade focus)
        "max_handle_depth": 0.18,   # 18% - CHANGED from 25% (tighter handles)
        "ideal_cup_depth": 0.15,    # NEW - 15% is sweet spot for 2-4 week targets
        "ideal_handle_depth": 0.10  # NEW - 10% handle indicates strong setup
    },
    "Inside Bar": {
        "tight_consolidation": 0.30,    # 30% of mother bar (daily)
        "tight_consolidation_weekly": 0.35,  # 35% for weekly
        "good_consolidation": 0.50,
        "good_consolidation_weekly": 0.55,
        "moderate_consolidation": 0.70,
        "moderate_consolidation_weekly": 0.75
    },
    "Inverse Head Shoulders": {
        "min_head_depth": 0.05,     # 5% minimum depth
        "max_head_depth": 0.40,     # 40% - CHANGED from 60% (swing trade focus)
        "min_symmetry": 0.40,       # 40% - CHANGED from 50% (more patterns, still reliable)
        "good_symmetry": 0.50,      # NEW - bonus threshold
        "excellent_symmetry": 0.70, # NEW - high confidence threshold
        "min_pattern_width_daily": 15,   # 15 days - CHANGED from 20 (catches faster patterns)
        "max_pattern_width_daily": 50,   # 50 days - CHANGED from 60 (swing focus)
        "min_pattern_width_weekly": 12,  # 12 weeks - CHANGED from 15
        "max_pattern_width_weekly": 35,  # 35 weeks - CHANGED from 40
        "impulsive_move_threshold": 0.6,  # 60% impulsive bars required
        "ideal_head_depth": 0.15,        # 15% ideal depth (good for 2-4 week targets)
        "ideal_head_depth_max": 0.20,    # NEW - upper bound of ideal range
        "pivot_strength_daily": 4,       # 4 - CHANGED from 5 (more flexible, still quality)
        "strict_pivot_strength_daily": 5, # NEW - bonus for perfect pivots
        "pivot_strength_weekly": 4,      # 4 (keep)
        "compact_pattern_days": 20,      # NEW - bonus if pattern forms quickly
        "extended_pattern_days": 40      # NEW - penalty if pattern takes too long
    }
}

# Chart Settings
CHART_CONFIG = {
    "height": 800,
    "volume_opacity": 0.7,
    "volume_colors": {
        "exceptional": "darkgreen",
        "strong": "green", 
        "good": "lightgreen",
        "weak": "red",
        "default": "blue"
    },
    "line_colors": {
        "entry": "green",
        "stop": "red",
        "target1": "lime",
        "target2": "darkgreen",
        "target3": "purple",
        "sma": "orange",
        "neckline": "blue",
        "left_shoulder": "cyan",
        "head": "magenta",
        "right_shoulder": "cyan"
    }
}

# Demo Data Settings (when yfinance unavailable)
DEMO_DATA_CONFIG = {
    "base_price_range": (50, 250),
    "volatility": 0.02,
    "volume_range": (1000000, 5000000)
}

# Export Settings
EXPORT_FILENAME_FORMAT = "patterns_v8_{timestamp}.csv"

# Error Messages
ERROR_MESSAGES = {
    "insufficient_data": "Insufficient data for analysis",
    "no_patterns": "No patterns detected. Try lowering confidence threshold.",
    "yfinance_unavailable": "Using demo data (yfinance not available)",
    "data_fetch_error": "Error fetching data, using demo data"
}

# Warning Messages
WARNING_MESSAGES = {
    "demo_mode": "Demo Mode: Using simulated data",
    "weekend_analysis": "Weekend Analysis: Patterns based on Friday's close",
    "friday_risk": "Friday entries require exceptional volume for weekend holds",
    "monday_gap": "Monday gap risk - validate patterns post-open"
}

# Success Messages
SUCCESS_MESSAGES = {
    "pattern_detected": "Pattern detected with high confidence",
    "volume_confirmed": "Volume confirmation present",
    "optimal_timing": "Optimal timing for entry"
}

# Disclaimer
DISCLAIMER_TEXT = """
DISCLAIMER: Educational purposes only. Not financial advice. 
Trading involves substantial risk. Consult professionals before trading.
"""
