# config.py
# Pattern Detector V8.2.2 - Configuration Settings
# CHATGPT-ALIGNED - Minimal IHS Rules for Swing Trading

# Application Settings
APP_TITLE = "Pro Pattern Detector v8.2.2"
APP_SUBTITLE = "ChatGPT-Aligned Swing Trading - Professional Pattern Recognition"
VERSION = "8.2.2"

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

# Pattern Age Limits (days/weeks) - CHATGPT-ALIGNED FOR SWING TRADING
PATTERN_AGE_LIMITS = {
    "daily": {
        "Flat Top Breakout": 8,
        "Bull Flag": 10,
        "Cup Handle": 35,
        "Inside Bar": 6,
        "Inverse Head Shoulders": 20   # CHANGED from 50 - ChatGPT: too stale for swings
    },
    "weekly": {
        "Flat Top Breakout": 8,
        "Bull Flag": 10,
        "Cup Handle": 25,
        "Inside Bar": 8,
        "Inverse Head Shoulders": 15   # CHANGED from 35 - swing-focused
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

# Pattern Detection Thresholds - CHATGPT-ALIGNED
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
        "min_cup_depth": 0.08,      # 8%
        "max_cup_depth": 0.30,      # 30% (swing-focused)
        "max_handle_depth": 0.18,   # 18% (tighter handles)
        "ideal_cup_depth": 0.15,    # 15% sweet spot
        "ideal_handle_depth": 0.10  # 10% ideal
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
        # CHATGPT-ALIGNED MINIMAL RULESET
        
        # Head depth requirements (ChatGPT: 4-6% minimum for daily)
        "min_head_depth": 0.04,              # 4% - CHANGED from 5% (ChatGPT recommendation)
        "max_head_depth": 0.40,              # 40% (keep - swing-focused)
        "ideal_head_depth_min": 0.05,        # 5% - NEW
        "ideal_head_depth_max": 0.20,        # 20% - ideal range
        
        # Shoulder symmetry (ChatGPT: ≤6-8% difference)
        "min_symmetry": 0.40,                # 40% minimum
        "good_symmetry": 0.50,               # 50% good
        "excellent_symmetry": 0.70,          # 70% excellent
        "max_shoulder_diff_pct": 0.08,       # 8% - NEW (ChatGPT: ≤6-8%)
        
        # Pattern width (ChatGPT: 12-35 days for daily)
        "min_pattern_width_daily": 12,       # 12 days - CHANGED from 15 (ChatGPT)
        "max_pattern_width_daily": 35,       # 35 days - CHANGED from 50 (ChatGPT)
        "min_pattern_width_weekly": 8,       # 8 weeks - CHANGED from 12
        "max_pattern_width_weekly": 20,      # 20 weeks - CHANGED from 35
        
        # Pivot strength (ChatGPT: N=2-3 for daily, simpler validation)
        "pivot_strength_daily": 3,           # 3 - CHANGED from 4 (N-left/N-right)
        "pivot_strength_weekly": 2,          # 2 - CHANGED from 4
        "strict_pivot_strength": 5,          # 5 - bonus for perfect pivots
        
        # Neckline validation (ChatGPT: ≤6-8% slope variance)
        "neckline_slope_max": 0.08,          # 8% - NEW (ChatGPT recommendation)
        "neckline_min_separation": 2,        # 2 bars - NEW (avoid single-bar spikes)
        
        # Breakout confirmation (ChatGPT: ≥0.5% above neckline)
        "breakout_buffer": 0.005,            # 0.5% - NEW (ChatGPT: must break above)
        "breakout_volume_min": 1.3,          # 1.3x - NEW (required for daily)
        
        # Freshness filter (ChatGPT: ≤5 bars since breakout)
        "freshness_bars_daily": 5,           # 5 bars - NEW (ChatGPT recommendation)
        "freshness_bars_weekly": 3,          # 3 bars - NEW
        
        # Pattern compactness bonuses/penalties
        "compact_pattern_days": 20,          # 20 days (bonus threshold)
        "extended_pattern_days": 30,         # 30 days - CHANGED from 40 (tighter)
        
        # REMOVED UNUSED FIELDS (ChatGPT: config should match reality)
        # "impulsive_move_threshold": DELETED - not validated in code
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
