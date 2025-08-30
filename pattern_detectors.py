# pattern_detectors.py
# Pattern Detector V8.1 - Pattern Detection Algorithms with Inverse Head & Shoulders

import numpy as np
from config import (
    PATTERN_THRESHOLDS, VOLUME_THRESHOLDS, VOLUME_SCORE_POINTS,
    PATTERN_VOLUME_BONUS, MAX_CONFIDENCE_WITHOUT_VOLUME,
    PATTERN_AGE_LIMITS, INSIDE_BAR_CONFIG
)

def analyze_volume_pattern(data, pattern_type, pattern_info):
    """Enhanced volume analysis with breakout confirmation and confidence capping"""
    volume_score = 0
    volume_info = {}
    
    if len(data) < 20:
        return volume_score, volume_info
    
    avg_volume_20 = data['Volume'].tail(20).mean()
    current_volume = data['Volume'].iloc[-1]
    recent_volume_5 = data['Volume'].tail(5).mean()
    
    volume_multiplier = current_volume / avg_volume_20
    recent_multiplier = recent_volume_5 / avg_volume_20
    
    volume_info['avg_volume_20'] = avg_volume_20
    volume_info['current_volume'] = current_volume
    volume_info['volume_multiplier'] = volume_multiplier
    volume_info['recent_multiplier'] = recent_multiplier
    
    # Score based on volume thresholds
    if volume_multiplier >= VOLUME_THRESHOLDS['exceptional']:
        volume_score += VOLUME_SCORE_POINTS['exceptional']
        volume_info['exceptional_volume'] = True
        volume_info['volume_status'] = f"Exceptional Volume ({volume_multiplier:.1f}x)"
    elif volume_multiplier >= VOLUME_THRESHOLDS['strong']:
        volume_score += VOLUME_SCORE_POINTS['strong']
        volume_info['strong_volume'] = True
        volume_info['volume_status'] = f"Strong Volume ({volume_multiplier:.1f}x)"
    elif volume_multiplier >= VOLUME_THRESHOLDS['good']:
        volume_score += VOLUME_SCORE_POINTS['good']
        volume_info['good_volume'] = True
        volume_info['volume_status'] = f"Good Volume ({volume_multiplier:.1f}x)"
    else:
        volume_info['weak_volume'] = True
        volume_info['volume_status'] = f"Weak Volume ({volume_multiplier:.1f}x)"
    
    # Pattern-specific volume analysis
    if pattern_type == "Bull Flag":
        if 'flagpole_gain' in pattern_info:
            try:
                flagpole_start = min(25, len(data) - 10)
                flagpole_end = 15
                
                flagpole_vol = data['Volume'].iloc[-flagpole_start:-flagpole_end].mean()
                flag_vol = data['Volume'].tail(15).mean()
                
                if flagpole_vol > flag_vol * 1.2:
                    volume_score += PATTERN_VOLUME_BONUS["Bull Flag"]
                    volume_info['flagpole_volume_pattern'] = True
                    volume_info['flagpole_vol_ratio'] = flagpole_vol / flag_vol
                elif flagpole_vol > flag_vol * 1.1:
                    volume_score += PATTERN_VOLUME_BONUS["Bull Flag"] // 2
                    volume_info['moderate_flagpole_volume'] = True
                    volume_info['flagpole_vol_ratio'] = flagpole_vol / flag_vol
            except:
                pass
    
    elif pattern_type == "Cup Handle":
        try:
            handle_days = min(30, len(data) // 3)
            if handle_days > 5:
                cup_data = data.iloc[:-handle_days]
                handle_data = data.tail(handle_days)
                
                if len(cup_data) > 10:
                    cup_volume = cup_data['Volume'].mean()
                    handle_volume = handle_data['Volume'].mean()
                    
                    if handle_volume < cup_volume * 0.80:
                        volume_score += PATTERN_VOLUME_BONUS["Cup Handle"]
                        volume_info['significant_volume_dryup'] = True
                        volume_info['handle_vol_ratio'] = handle_volume / cup_volume
                    elif handle_volume < cup_volume * 0.90:
                        volume_score += PATTERN_VOLUME_BONUS["Cup Handle"] * 0.75
                        volume_info['moderate_volume_dryup'] = True
                        volume_info['handle_vol_ratio'] = handle_volume / cup_volume
        except:
            pass
    
    elif pattern_type == "Flat Top Breakout":
        resistance_tests = data['Volume'].tail(20)
        avg_resistance_volume = resistance_tests.mean()
        
        if current_volume > avg_resistance_volume * 1.4:
            volume_score += PATTERN_VOLUME_BONUS["Flat Top Breakout"]
            volume_info['breakout_volume_surge'] = True
            volume_info['resistance_vol_ratio'] = current_volume / avg_resistance_volume
        elif current_volume > avg_resistance_volume * 1.2:
            volume_score += PATTERN_VOLUME_BONUS["Flat Top Breakout"] * 0.75
            volume_info['moderate_breakout_volume'] = True
            volume_info['resistance_vol_ratio'] = current_volume / avg_resistance_volume
    
    elif pattern_type == "Inside Bar":
        # Prefer lower volume during consolidation
        if volume_multiplier < 0.8:
            volume_score += PATTERN_VOLUME_BONUS["Inside Bar"]
            volume_info['consolidation_volume'] = True
        elif volume_multiplier < 1.0:
            volume_score += PATTERN_VOLUME_BONUS["Inside Bar"] * 0.67
            volume_info['quiet_consolidation'] = True
        
        # Check for volume expansion on potential breakout
        if volume_multiplier >= 1.5:
            volume_score += PATTERN_VOLUME_BONUS["Inside Bar"]
            volume_info['breakout_volume_expansion'] = True
    
    # Volume trend analysis
    volume_trend = data['Volume'].tail(5).mean() / data['Volume'].tail(20).mean()
    if volume_trend > 1.1:
        volume_score += 5
        volume_info['increasing_volume_trend'] = True
    elif volume_trend < 0.9:
        volume_score += 5
        volume_info['decreasing_volume_trend'] = True
    
    return volume_score, volume_info

def detect_cup_handle(data, macd_line, signal_line, histogram, market_context):
    """Detect cup handle with enhanced volume analysis"""
    confidence = 0
    pattern_info = {}
    
    if len(data) < 30:
        return confidence, pattern_info
    
    thresholds = PATTERN_THRESHOLDS["Cup Handle"]
    
    max_lookback = min(100, len(data) - 3)
    handle_days = min(30, max_lookback // 3)
    
    cup_data = data.iloc[-max_lookback:-handle_days] if handle_days > 0 else data.iloc[-max_lookback:]
    handle_data = data.tail(handle_days) if handle_days > 0 else data.tail(5)
    
    if len(cup_data) < 15:
        return confidence, pattern_info
    
    cup_start = cup_data['Close'].iloc[0]
    cup_bottom = cup_data['Low'].min()
    cup_right = cup_data['Close'].iloc[-1]
    cup_depth = (max(cup_start, cup_right) - cup_bottom) / max(cup_start, cup_right)
    
    if cup_depth < thresholds['min_cup_depth'] or cup_depth > thresholds['max_cup_depth']:
        return confidence, pattern_info
    
    if cup_right < cup_start * 0.75:
        return confidence, pattern_info
    
    confidence += 25
    pattern_info['cup_depth'] = f"{cup_depth*100:.1f}%"
    
    # Handle analysis
    if handle_days > 0:
        handle_low = handle_data['Low'].min()
        current_price = data['Close'].iloc[-1]
        handle_depth = (cup_right - handle_low) / cup_right
        
        if handle_depth > thresholds['max_handle_depth']:
            confidence += 10
            pattern_info['deep_handle'] = f"{handle_depth*100:.1f}%"
        elif handle_depth <= 0.08:
            confidence += 20
            pattern_info['perfect_handle'] = f"{handle_depth*100:.1f}%"
        elif handle_depth <= 0.15:
            confidence += 15
            pattern_info['good_handle'] = f"{handle_depth*100:.1f}%"
        else:
            confidence += 10
            pattern_info['acceptable_handle'] = f"{handle_depth*100:.1f}%"
        
        # Handle length analysis
        if handle_days > 25:
            confidence *= 0.8
            pattern_info['long_handle'] = f"{handle_days} days"
        elif handle_days <= 10:
            confidence += 10
            pattern_info['short_handle'] = f"{handle_days} days"
        elif handle_days <= 20:
            confidence += 5
            pattern_info['medium_handle'] = f"{handle_days} days"
    else:
        confidence += 10
        pattern_info['forming_handle'] = "Handle forming"
    
    # Position analysis
    current_price = data['Close'].iloc[-1]
    breakout_level = max(cup_start, cup_right)
    
    if current_price < breakout_level * 0.70:
        confidence *= 0.7
        pattern_info['far_from_rim'] = True
    else:
        confidence += 5
    
    if handle_days > 0:
        handle_low = handle_data['Low'].min()
        if current_price < handle_low * 0.90:
            confidence *= 0.8
            pattern_info['below_handle'] = True
    
    # Technical confirmation
    if macd_line.iloc[-1] > signal_line.iloc[-1]:
        confidence += 10
        pattern_info['macd_bullish'] = True
    
    # Volume analysis
    volume_score, volume_info = analyze_volume_pattern(data, "Cup Handle", pattern_info)
    confidence += volume_score
    pattern_info.update(volume_info)
    
    if confidence < 35:
        return confidence, pattern_info
    
    # Apply volume confirmation cap
    if not (volume_info.get('good_volume') or volume_info.get('strong_volume') or volume_info.get('exceptional_volume')):
        confidence = min(confidence, MAX_CONFIDENCE_WITHOUT_VOLUME)
        pattern_info['confidence_capped'] = "No volume confirmation"
    
    return confidence, pattern_info

def detect_inverse_head_shoulders(data, macd_line, signal_line, histogram, market_context, timeframe="daily"):
    """
    Enhanced Inverse Head and Shoulders Pattern Detection
    
    Combines best practices from:
    1. TradingView's 5/5 pivot system with symmetry validation
    2. Academic research emphasizing volume confirmation
    3. Professional trader requirements for shoulder alignment
    
    Unique enhancements:
    - Multi-timeframe pivot validation
    - Advanced symmetry scoring system
    - Impulsive move validation
    - Neckline slope analysis
    - Volume distribution analysis
    """
    confidence = 0
    pattern_info = {}
    
    if len(data) < 60:
        return confidence, pattern_info
    
    thresholds = PATTERN_THRESHOLDS["Inverse Head Shoulders"]
    
    # Adjust parameters based on timeframe
    if timeframe == "1wk":
        min_pattern_bars = 15  # weeks
        pivot_strength = thresholds['pivot_strength_weekly']
        aging_threshold = PATTERN_AGE_LIMITS['weekly']['Inverse Head Shoulders']
        pattern_info['timeframe'] = 'Weekly'
        min_width = thresholds['min_pattern_width_weekly']
        max_width = thresholds['max_pattern_width_weekly']
    else:
        min_pattern_bars = 40  # days
        pivot_strength = thresholds['pivot_strength_daily']
        aging_threshold = PATTERN_AGE_LIMITS['daily']['Inverse Head Shoulders']
        pattern_info['timeframe'] = 'Daily'
        min_width = thresholds['min_pattern_width_daily']
        max_width = thresholds['max_pattern_width_daily']
    
    if len(data) < min_pattern_bars:
        return confidence, pattern_info
    
    # Step 1: Find pivot lows (inverse head and shoulders uses lows)
    pivot_lows = find_pivot_lows(data, pivot_strength, min_pattern_bars)
    
    if len(pivot_lows) < 3:
        return confidence, pattern_info
    
    # Step 2: Find valid inverse head and shoulders formations
    valid_patterns = find_inverse_head_shoulders_formations(data, pivot_lows, timeframe, min_width, max_width, thresholds)
    
    if not valid_patterns:
        return confidence, pattern_info
    
    # Step 3: Score and select best pattern
    best_pattern = select_best_pattern(valid_patterns, data)
    
    if not best_pattern:
        return confidence, pattern_info
    
    # Step 4: Validate pattern quality and calculate confidence
    confidence, pattern_info = validate_and_score_ihs_pattern(
        data, best_pattern, macd_line, signal_line, histogram, 
        market_context, timeframe, aging_threshold, thresholds
    )
    
    return confidence, pattern_info

def find_pivot_lows(data, pivot_strength, lookback_period):
    """Find pivot lows using TradingView's proven 5/5 system"""
    pivot_lows = []
    
    # Only look at recent data for pattern formation
    start_idx = max(0, len(data) - lookback_period)
    search_data = data.iloc[start_idx:]
    
    for i in range(pivot_strength, len(search_data) - pivot_strength):
        current_low = search_data['Low'].iloc[i]
        current_idx = start_idx + i
        
        # Check left bars
        left_valid = True
        for j in range(1, pivot_strength + 1):
            if search_data['Low'].iloc[i - j] <= current_low:
                left_valid = False
                break
        
        # Check right bars
        right_valid = True
        for j in range(1, pivot_strength + 1):
            if search_data['Low'].iloc[i + j] <= current_low:
                right_valid = False
                break
        
        if left_valid and right_valid:
            pivot_lows.append({
                'idx': current_idx,
                'price': current_low,
                'date': search_data.index[i]
            })
    
    return pivot_lows

def find_inverse_head_shoulders_formations(data, pivot_lows, timeframe, min_width, max_width, thresholds):
    """Find valid inverse H&S patterns from pivot lows"""
    valid_patterns = []
    
    # Need at least 3 pivot lows
    if len(pivot_lows) < 3:
        return valid_patterns
    
    # Test all combinations of 3 consecutive pivot lows
    for i in range(len(pivot_lows) - 2):
        left_shoulder = pivot_lows[i]
        head = pivot_lows[i + 1]
        right_shoulder = pivot_lows[i + 2]
        
        # Basic inverse H&S structure: head must be lowest
        if not (head['price'] < left_shoulder['price'] and head['price'] < right_shoulder['price']):
            continue
        
        # Pattern width validation
        pattern_width = right_shoulder['idx'] - left_shoulder['idx']
        if not (min_width <= pattern_width <= max_width):
            continue
        
        # Find neckline points (highs between the lows)
        left_neck_point = find_peak_between(data, left_shoulder['idx'], head['idx'])
        right_neck_point = find_peak_between(data, head['idx'], right_shoulder['idx'])
        
        if not left_neck_point or not right_neck_point:
            continue
        
        # Calculate pattern metrics
        head_depth = calculate_head_depth(left_shoulder, head, right_shoulder)
        shoulder_symmetry = calculate_shoulder_symmetry(left_shoulder, head, right_shoulder)
        
        # Validate head depth
        if not (thresholds['min_head_depth'] <= head_depth <= thresholds['max_head_depth']):
            continue
        
        # Validate symmetry
        if shoulder_symmetry < thresholds['min_symmetry']:
            continue
        
        pattern = {
            'left_shoulder': left_shoulder,
            'head': head,
            'right_shoulder': right_shoulder,
            'left_neck_point': left_neck_point,
            'right_neck_point': right_neck_point,
            'neckline_slope': calculate_neckline_slope(left_neck_point, right_neck_point),
            'pattern_width': pattern_width,
            'head_depth': head_depth,
            'shoulder_symmetry': shoulder_symmetry
        }
        
        # Advanced validation
        if validate_pattern_structure(data, pattern, timeframe, thresholds):
            valid_patterns.append(pattern)
    
    return valid_patterns

def find_peak_between(data, start_idx, end_idx):
    """Find the highest point between two indices"""
    if start_idx >= end_idx or end_idx >= len(data):
        return None
    
    segment = data.iloc[start_idx:end_idx + 1]
    if len(segment) == 0:
        return None
    
    max_idx = segment['High'].idxmax()
    max_price = segment['High'].max()
    
    return {
        'idx': data.index.get_loc(max_idx),
        'price': max_price,
        'date': max_idx
    }

def calculate_neckline_slope(left_point, right_point):
    """Calculate neckline slope - negative is preferred for inverse H&S"""
    if not left_point or not right_point:
        return 0
    
    price_diff = right_point['price'] - left_point['price']
    time_diff = right_point['idx'] - left_point['idx']
    
    if time_diff == 0:
        return 0
    
    return price_diff / time_diff

def calculate_head_depth(left_shoulder, head, right_shoulder):
    """Calculate how deep the head is relative to shoulders"""
    avg_shoulder_price = (left_shoulder['price'] + right_shoulder['price']) / 2
    depth_ratio = (avg_shoulder_price - head['price']) / avg_shoulder_price
    return depth_ratio

def calculate_shoulder_symmetry(left_shoulder, head, right_shoulder):
    """Calculate shoulder symmetry score (0-1, higher is better)"""
    # Time symmetry
    left_time = head['idx'] - left_shoulder['idx']
    right_time = right_shoulder['idx'] - head['idx']
    
    if left_time == 0 or right_time == 0:
        time_symmetry = 0
    else:
        time_ratio = min(left_time, right_time) / max(left_time, right_time)
        time_symmetry = time_ratio
    
    # Price symmetry (shoulders should be at similar levels)
    price_diff = abs(left_shoulder['price'] - right_shoulder['price'])
    avg_price = (left_shoulder['price'] + right_shoulder['price']) / 2
    
    if avg_price == 0:
        price_symmetry = 0
    else:
        price_symmetry = max(0, 1 - (price_diff / avg_price) * 2)  # Penalize price differences
    
    # Combined symmetry score
    combined_symmetry = (time_symmetry * 0.6 + price_symmetry * 0.4)
    return combined_symmetry

def validate_pattern_structure(data, pattern, timeframe, thresholds):
    """Advanced pattern structure validation"""
    
    # Validate impulsive moves (key professional requirement)
    if not validate_impulsive_moves(data, pattern, thresholds['impulsive_move_threshold']):
        return False
    
    return True

def validate_impulsive_moves(data, pattern, threshold):
    """Ensure moves are impulsive, not choppy (professional requirement)"""
    
    # Check left shoulder to head move
    left_move_data = data.iloc[pattern['left_shoulder']['idx']:pattern['head']['idx'] + 1]
    if not is_impulsive_move(left_move_data, 'down', threshold):
        return False
    
    # Check head to right shoulder move
    right_move_data = data.iloc[pattern['head']['idx']:pattern['right_shoulder']['idx'] + 1]
    if not is_impulsive_move(right_move_data, 'up', threshold):
        return False
    
    return True

def is_impulsive_move(price_data, direction, threshold):
    """Check if price movement is impulsive rather than choppy"""
    if len(price_data) < 3:
        return True
    
    if direction == 'down':
        # For downward moves, check if predominantly declining
        declining_bars = sum(1 for i in range(1, len(price_data)) 
                           if price_data['Close'].iloc[i] < price_data['Close'].iloc[i-1])
        impulsive_ratio = declining_bars / (len(price_data) - 1)
        return impulsive_ratio >= threshold
    
    else:  # direction == 'up'
        # For upward moves, check if predominantly rising
        rising_bars = sum(1 for i in range(1, len(price_data)) 
                        if price_data['Close'].iloc[i] > price_data['Close'].iloc[i-1])
        impulsive_ratio = rising_bars / (len(price_data) - 1)
        return impulsive_ratio >= threshold

def select_best_pattern(valid_patterns, data):
    """Select the highest quality pattern using scoring system"""
    if not valid_patterns:
        return None
    
    best_pattern = None
    best_score = 0
    
    for pattern in valid_patterns:
        score = 0
        
        # Score symmetry (25 points max)
        score += pattern['shoulder_symmetry'] * 25
        
        # Score head depth (20 points max) - prefer moderate depth
        ideal_depth = PATTERN_THRESHOLDS["Inverse Head Shoulders"]['ideal_head_depth']
        depth_score = 20 * (1 - abs(pattern['head_depth'] - ideal_depth) / ideal_depth)
        score += max(0, min(20, depth_score))
        
        # Score neckline slope (15 points max) - prefer downward slope
        if pattern['neckline_slope'] < 0:
            score += 15  # Downward slope is ideal
        elif pattern['neckline_slope'] == 0:
            score += 10  # Flat is acceptable
        else:
            score += 5   # Upward slope is less ideal
        
        # Score recency (10 points max) - prefer more recent patterns
        current_idx = len(data) - 1
        recency_score = 10 * (1 - (current_idx - pattern['right_shoulder']['idx']) / 30)
        score += max(0, recency_score)
        
        if score > best_score:
            best_score = score
            best_pattern = pattern
    
    return best_pattern

def validate_and_score_ihs_pattern(data, pattern, macd_line, signal_line, histogram, 
                                 market_context, timeframe, aging_threshold, thresholds):
    """Final validation and confidence scoring for inverse head and shoulders"""
    confidence = 0
    pattern_info = {}
    
    # Base confidence for valid pattern structure
    confidence += 30
    
    # Add pattern-specific information
    pattern_info.update({
        'left_shoulder_price': pattern['left_shoulder']['price'],
        'head_price': pattern['head']['price'],
        'right_shoulder_price': pattern['right_shoulder']['price'],
        'left_neck_price': pattern['left_neck_point']['price'],
        'right_neck_price': pattern['right_neck_point']['price'],
        'neckline_slope': pattern['neckline_slope'],
        'head_depth_percent': f"{pattern['head_depth']*100:.1f}%",
        'shoulder_symmetry_score': f"{pattern['shoulder_symmetry']:.2f}",
        'pattern_width_bars': pattern['pattern_width']
    })
    
    # Symmetry bonus (inspired by professional traders' emphasis)
    if pattern['shoulder_symmetry'] > 0.8:
        confidence += 20
        pattern_info['excellent_symmetry'] = True
    elif pattern['shoulder_symmetry'] > 0.7:
        confidence += 15
        pattern_info['good_symmetry'] = True
    elif pattern['shoulder_symmetry'] > 0.6:
        confidence += 10
        pattern_info['fair_symmetry'] = True
    
    # Head depth scoring
    ideal_depth = thresholds['ideal_head_depth']
    if abs(pattern['head_depth'] - ideal_depth) < 0.05:
        confidence += 15
        pattern_info['optimal_head_depth'] = True
    elif 0.10 <= pattern['head_depth'] <= 0.25:
        confidence += 10
        pattern_info['acceptable_head_depth'] = True
    
    # Neckline slope bonus (research shows downward slope is preferred)
    if pattern['neckline_slope'] < -0.01:
        confidence += 15
        pattern_info['ideal_downward_neckline'] = True
    elif pattern['neckline_slope'] < 0:
        confidence += 10
        pattern_info['good_downward_neckline'] = True
    elif abs(pattern['neckline_slope']) < 0.01:
        confidence += 5
        pattern_info['flat_neckline'] = True
    
    # Current position analysis
    current_price = data['Close'].iloc[-1]
    neckline_price = pattern['right_neck_point']['price']
    
    # Check if near breakout
    distance_to_neckline = (neckline_price - current_price) / current_price
    
    if distance_to_neckline < 0.02:  # Within 2% of neckline
        confidence += 15
        pattern_info['near_breakout'] = True
        pattern_info['distance_to_neckline'] = f"{distance_to_neckline*100:.1f}%"
    elif distance_to_neckline < 0.05:  # Within 5% of neckline
        confidence += 10
        pattern_info['approaching_neckline'] = True
    
    # Technical confirmation
    if macd_line.iloc[-1] > signal_line.iloc[-1]:
        confidence += 10
        pattern_info['macd_bullish'] = True
    
    if histogram.iloc[-1] > histogram.iloc[-3]:
        confidence += 10
        pattern_info['momentum_improving'] = True
    
    # Enhanced volume analysis
    volume_score, volume_info = analyze_inverse_head_shoulders_volume(data, pattern)
    confidence += volume_score
    pattern_info.update(volume_info)
    
    # Apply volume confirmation cap (research finding)
    if not (volume_info.get('volume_pattern_confirmed') or 
            volume_info.get('strong_volume') or 
            volume_info.get('exceptional_volume')):
        confidence = min(confidence, MAX_CONFIDENCE_WITHOUT_VOLUME)
        pattern_info['confidence_capped'] = "No volume confirmation"
    
    # Age penalty
    bars_since_right_shoulder = len(data) - 1 - pattern['right_shoulder']['idx']
    if bars_since_right_shoulder > aging_threshold:
        confidence *= 0.7
        pattern_info['pattern_aging'] = True
        pattern_info['age_bars'] = bars_since_right_shoulder
    
    # Invalidation checks
    invalidation_issues = check_pattern_invalidation(data, pattern)
    if invalidation_issues:
        confidence *= 0.5
        pattern_info.update(invalidation_issues)
    
    return confidence, pattern_info

def analyze_inverse_head_shoulders_volume(data, pattern):
    """Advanced volume analysis specific to inverse head and shoulders"""
    volume_score = 0
    volume_info = {}
    
    if len(data) < 20:
        return volume_score, volume_info
    
    # Get volume data for each phase
    left_shoulder_vol = data['Volume'].iloc[pattern['left_shoulder']['idx']]
    head_vol = data['Volume'].iloc[pattern['head']['idx']]
    right_shoulder_vol = data['Volume'].iloc[pattern['right_shoulder']['idx']]
    current_vol = data['Volume'].iloc[-1]
    avg_volume = data['Volume'].tail(20).mean()
    
    # Classic inverse H&S volume pattern: decreasing from left to right shoulder
    if left_shoulder_vol > head_vol > right_shoulder_vol:
        volume_score += 20
        volume_info['classic_volume_pattern'] = True
        volume_info['volume_pattern_confirmed'] = True
    elif left_shoulder_vol > right_shoulder_vol:
        volume_score += 10
        volume_info['diminishing_volume_pattern'] = True
    
    # Current volume analysis
    current_multiplier = current_vol / avg_volume
    volume_info['current_volume_ratio'] = f"{current_multiplier:.1f}x"
    
    if current_multiplier >= 2.0:
        volume_score += 25
        volume_info['exceptional_volume'] = True
        volume_info['volume_status'] = f"Exceptional Volume ({current_multiplier:.1f}x)"
    elif current_multiplier >= 1.5:
        volume_score += 20
        volume_info['strong_volume'] = True
        volume_info['volume_status'] = f"Strong Volume ({current_multiplier:.1f}x)"
    elif current_multiplier >= 1.3:
        volume_score += 15
        volume_info['good_volume'] = True
        volume_info['volume_status'] = f"Good Volume ({current_multiplier:.1f}x)"
    else:
        volume_info['volume_status'] = f"Weak Volume ({current_multiplier:.1f}x)"
    
    # Volume trend analysis from right shoulder to current
    rs_idx = pattern['right_shoulder']['idx']
    if len(data) - rs_idx > 3:
        recent_volume_trend = data['Volume'].iloc[rs_idx:].mean() / data['Volume'].iloc[rs_idx-10:rs_idx].mean()
        if recent_volume_trend > 1.2:
            volume_score += 10
            volume_info['increasing_volume_trend'] = True
    
    return volume_score, volume_info

def check_pattern_invalidation(data, pattern):
    """Check for pattern invalidation conditions"""
    issues = {}
    current_price = data['Close'].iloc[-1]
    
    # Check if price has fallen below head (major invalidation)
    if current_price < pattern['head']['price'] * 0.98:
        issues['major_invalidation'] = "Price below head level"
        issues['pattern_broken'] = True
    
    # Check if price is far below right shoulder
    elif current_price < pattern['right_shoulder']['price'] * 0.95:
        issues['minor_invalidation'] = "Price significantly below right shoulder"
    
    return issues

def detect_pattern(data, pattern_type, market_context, timeframe="daily"):
    """Main pattern detection function with enhanced volume analysis and timing awareness"""
    if len(data) < 10:
        return False, 0, {}
    
    # Calculate MACD components
    ema_fast = data['Close'].ewm(span=12).mean()
    ema_slow = data['Close'].ewm(span=26).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=9).mean()
    histogram = macd_line - signal_line
    
    confidence = 0
    pattern_info = {}
    
    # Route to appropriate pattern detector
    if pattern_type == "Flat Top Breakout":
        confidence, pattern_info = detect_flat_top(data, macd_line, signal_line, histogram, market_context)
        confidence = min(confidence, 100)
        
    elif pattern_type == "Bull Flag":
        confidence, pattern_info = detect_bull_flag(data, macd_line, signal_line, histogram, market_context)
        confidence = min(confidence * 1.05, 100)
        
    elif pattern_type == "Cup Handle":
        confidence, pattern_info = detect_cup_handle(data, macd_line, signal_line, histogram, market_context)
        confidence = min(confidence * 1.1, 100)
        
    elif pattern_type == "Inside Bar":
        confidence, pattern_info = detect_inside_bar(data, macd_line, signal_line, histogram, market_context, timeframe)
        confidence = min(confidence, 100)
        
    elif pattern_type == "Inverse Head Shoulders":
        confidence, pattern_info = detect_inverse_head_shoulders(data, macd_line, signal_line, histogram, market_context, timeframe)
        confidence = min(confidence, 100)
    
    # Add MACD data to pattern info for charting
    pattern_info['macd_line'] = macd_line
    pattern_info['signal_line'] = signal_line
    pattern_info['histogram'] = histogram
    
    # Apply market timing adjustments (implemented in market_timing module)
    
    return confidence >= 55, confidence, pattern_info_inside_bar(data, macd_line, signal_line, histogram, market_context, timeframe="daily"):
    """Detect Inside Bar pattern - buy-only with specific entry rules and color requirements"""
    confidence = 0
    pattern_info = {}
    
    if len(data) < 5:
        return confidence, pattern_info
    
    # Adjust lookback based on timeframe
    if timeframe == "1wk":
        max_lookback_range = range(-1, -7, -1)  # Look back 6 weeks maximum
        aging_threshold = -8  # Pattern stale after 8 weeks
        pattern_info['timeframe'] = 'Weekly'
    else:
        max_lookback_range = range(-1, -5, -1)  # Look back 4 days maximum  
        aging_threshold = -6  # Pattern stale after 6 days
        pattern_info['timeframe'] = 'Daily'
    
    # Look for inside bar pattern (max 2 inside bars)
    mother_bar_idx = None
    inside_bars_count = 0
    inside_bar_indices = []
    
    # Start from the most recent bar and look backwards
    for i in max_lookback_range:
        try:
            current_bar = data.iloc[i]
            previous_bar = data.iloc[i-1]
            
            # Check if current bar is inside previous bar
            is_inside = (current_bar['High'] <= previous_bar['High'] and 
                        current_bar['Low'] >= previous_bar['Low'] and
                        current_bar['High'] < previous_bar['High'] and  # Must be strictly inside
                        current_bar['Low'] > previous_bar['Low'])
            
            # Color validation: Mother bar must be green, inside bar must be red
            mother_is_green = previous_bar['Close'] > previous_bar['Open']
            inside_is_red = current_bar['Close'] < current_bar['Open']
            
            if is_inside and mother_is_green and inside_is_red:
                if inside_bars_count == 0:
                    # First inside bar found, previous bar is mother bar
                    mother_bar_idx = i - 1
                    inside_bar_indices.append(i)
                    inside_bars_count = 1
                elif inside_bars_count == 1 and i == inside_bar_indices[0] - 1:
                    # Second consecutive inside bar (must also be red)
                    inside_bar_indices.append(i)
                    inside_bars_count = 2
                    break  # Max 2 inside bars
                else:
                    break  # Not consecutive, stop looking
            else:
                break  # No valid inside bar (size or color), stop looking
        except (IndexError, KeyError):
            break
    
    if inside_bars_count == 0:
        return confidence, pattern_info
    
    # Get mother bar and inside bar(s) data
    mother_bar = data.iloc[mother_bar_idx]
    latest_inside_bar = data.iloc[inside_bar_indices[0]]  # Most recent inside bar
    
    # Validate color requirements one more time
    mother_is_green = mother_bar['Close'] > mother_bar['Open']
    inside_is_red = latest_inside_bar['Close'] < latest_inside_bar['Open']
    
    if not (mother_is_green and inside_is_red):
        return confidence, pattern_info
    
    # Base confidence for pattern formation
    base_confidence = 35 if timeframe == "1wk" else 30  # Higher base for weekly patterns
    confidence += base_confidence
    
    pattern_info['mother_bar_high'] = mother_bar['High']
    pattern_info['mother_bar_low'] = mother_bar['Low']
    pattern_info['inside_bar_high'] = latest_inside_bar['High']
    pattern_info['inside_bar_low'] = latest_inside_bar['Low']
    pattern_info['inside_bars_count'] = inside_bars_count
    pattern_info['color_validated'] = True
    pattern_info['mother_bar_color'] = 'Green'
    pattern_info['inside_bar_color'] = 'Red'
    
    # Bonus for proper color combination
    confidence += 15
    pattern_info['proper_color_combo'] = True
    
    # Prefer single inside bar over double
    if inside_bars_count == 1:
        confidence += 15
        pattern_info['single_inside_bar'] = True
    else:
        confidence += 10
        pattern_info['double_inside_bar'] = True
    
    # Calculate inside bar size relative to mother bar
    mother_bar_range = mother_bar['High'] - mother_bar['Low']
    inside_bar_range = latest_inside_bar['High'] - latest_inside_bar['Low']
    
    if mother_bar_range > 0:
        size_ratio = inside_bar_range / mother_bar_range
        pattern_info['size_ratio'] = f"{size_ratio:.1%}"
        
        # Get thresholds based on timeframe
        thresholds = PATTERN_THRESHOLDS["Inside Bar"]
        tight_threshold = thresholds['tight_consolidation_weekly'] if timeframe == "1wk" else thresholds['tight_consolidation']
        good_threshold = thresholds['good_consolidation_weekly'] if timeframe == "1wk" else thresholds['good_consolidation']
        moderate_threshold = thresholds['moderate_consolidation_weekly'] if timeframe == "1wk" else thresholds['moderate_consolidation']
        
        if size_ratio < tight_threshold:
            confidence += 20
            pattern_info['tight_consolidation'] = True
        elif size_ratio < good_threshold:
            confidence += 15
            pattern_info['good_consolidation'] = True
        elif size_ratio < moderate_threshold:
            confidence += 10
            pattern_info['moderate_consolidation'] = True
        else:
            confidence += 5
    
    # Check position within mother bar (prefer middle positioning)
    mother_bar_midpoint = (mother_bar['High'] + mother_bar['Low']) / 2
    inside_bar_midpoint = (latest_inside_bar['High'] + latest_inside_bar['Low']) / 2
    
    distance_from_middle = abs(inside_bar_midpoint - mother_bar_midpoint) / mother_bar_range
    if distance_from_middle < 0.25:
        confidence += 10
        pattern_info['centered_inside_bar'] = True
    elif distance_from_middle < 0.35:
        confidence += 5
        pattern_info['well_positioned'] = True
    
    # Technical confirmation
    if macd_line.iloc[-1] > signal_line.iloc[-1]:
        confidence += 15
        pattern_info['macd_bullish'] = True
    
    if histogram.iloc[-1] > histogram.iloc[-3]:
        confidence += 10
        pattern_info['momentum_improving'] = True
    
    # Current price should be near inside bar range for valid setup
    current_price = data['Close'].iloc[-1]
    if current_price >= latest_inside_bar['Low'] * 0.98:
        confidence += 10
        pattern_info['price_in_range'] = True
    
    # Volume analysis
    volume_score, volume_info = analyze_volume_pattern(data, "Inside Bar", pattern_info)
    confidence += volume_score
    pattern_info.update(volume_info)
    
    # Apply volume confirmation cap
    if not (volume_info.get('good_volume') or volume_info.get('strong_volume') or volume_info.get('exceptional_volume')):
        confidence = min(confidence, MAX_CONFIDENCE_WITHOUT_VOLUME)
        pattern_info['confidence_capped'] = "No volume confirmation"
    
    # Pattern age check - timeframe adjusted
    if mother_bar_idx <= aging_threshold:
        aging_penalty = 0.7 if timeframe == "1wk" else 0.8
        confidence *= aging_penalty
        pattern_info['pattern_aging'] = True
        pattern_info['age_periods'] = abs(mother_bar_idx)
    
    return confidence, pattern_info

def detect_flat_top(data, macd_line, signal_line, histogram, market_context):
    """Detect flat top with enhanced volume"""
    confidence = 0
    pattern_info = {}
    
    if len(data) < 50:
        return confidence, pattern_info
    
    thresholds = PATTERN_THRESHOLDS["Flat Top Breakout"]
    
    ascent_start = min(45, len(data) - 15)
    ascent_end = 25
    
    start_price = data['Close'].iloc[-ascent_start]
    peak_price = data['High'].iloc[-ascent_start:-ascent_end].max()
    initial_gain = (peak_price - start_price) / start_price
    
    if initial_gain < thresholds['min_initial_gain']:
        return confidence, pattern_info
    
    confidence += 25
    pattern_info['initial_ascension'] = f"{initial_gain*100:.1f}%"
    
    descent_data = data.iloc[-ascent_end:-10]
    descent_low = descent_data['Low'].min()
    pullback = (peak_price - descent_low) / peak_price
    
    if pullback < thresholds['min_pullback']:
        return confidence, pattern_info
    
    # Check for descending highs
    descent_highs = descent_data['High'].rolling(3, center=True).max().dropna()
    if len(descent_highs) >= 2:
        if descent_highs.iloc[-1] < descent_highs.iloc[0] * 0.97:
            confidence += 20
            pattern_info['descending_highs'] = True
    
    # Check for higher lows
    current_lows = data.tail(15)['Low'].rolling(3, center=True).min().dropna()
    if len(current_lows) >= 3:
        if current_lows.iloc[-1] > current_lows.iloc[0] * 1.01:
            confidence += 25
            pattern_info['higher_lows'] = True
    
    resistance_level = peak_price
    touches = sum(1 for h in data['High'].tail(20) if h >= resistance_level * thresholds['resistance_tolerance'])
    if touches >= 2:
        confidence += 15
        pattern_info['resistance_level'] = resistance_level
        pattern_info['resistance_touches'] = touches
    
    # Age and invalidation checks
    current_price = data['Close'].iloc[-1]
    days_old = next((i for i in range(1, 11) if data['High'].iloc[-i] >= resistance_level * thresholds['resistance_tolerance']), 11)
    
    if days_old > PATTERN_AGE_LIMITS['daily']['Flat Top Breakout']:
        return confidence * 0.5, {**pattern_info, 'pattern_stale': True, 'days_old': days_old}
    
    if current_price < descent_low * 0.95:
        return 0, {'pattern_broken': True, 'break_reason': 'Below support'}
    
    # Technical confirmation
    if macd_line.iloc[-1] > signal_line.iloc[-1]:
        confidence += 10
        pattern_info['macd_bullish'] = True
    
    # Volume analysis
    volume_score, volume_info = analyze_volume_pattern(data, "Flat Top Breakout", pattern_info)
    confidence += volume_score
    pattern_info.update(volume_info)
    
    # Apply volume confirmation cap
    if not (volume_info.get('good_volume') or volume_info.get('strong_volume') or volume_info.get('exceptional_volume')):
        confidence = min(confidence, MAX_CONFIDENCE_WITHOUT_VOLUME)
        pattern_info['confidence_capped'] = "No volume confirmation"
    
    return confidence, pattern_info

def detect_bull_flag(data, macd_line, signal_line, histogram, market_context):
    """Detect bull flag with enhanced volume analysis"""
    confidence = 0
    pattern_info = {}
    
    if len(data) < 30:
        return confidence, pattern_info
    
    thresholds = PATTERN_THRESHOLDS["Bull Flag"]
    
    flagpole_start = min(25, len(data) - 10)
    flagpole_end = 15
    
    start_price = data['Close'].iloc[-flagpole_start]
    peak_price = data['High'].iloc[-flagpole_start:-flagpole_end].max()
    flagpole_gain = (peak_price - start_price) / start_price
    
    if flagpole_gain < thresholds['min_flagpole_gain']:
        return confidence, pattern_info
    
    confidence += 25
    pattern_info['flagpole_gain'] = f"{flagpole_gain*100:.1f}%"
    
    flag_data = data.tail(15)
    flag_start = data['Close'].iloc[-flagpole_end]
    current_price = data['Close'].iloc[-1]
    
    pullback = (current_price - flag_start) / flag_start
    pullback_range = thresholds['pullback_range']
    if pullback_range[0] <= pullback <= pullback_range[1]:
        confidence += 20
        pattern_info['flag_pullback'] = f"{pullback*100:.1f}%"
        pattern_info['healthy_pullback'] = True
    
    # Invalidation checks
    flag_low = flag_data['Low'].min()
    if current_price < flag_low * thresholds['flag_tolerance']:
        return 0, {'pattern_broken': True, 'break_reason': 'Below flag support'}
    
    if current_price < start_price:
        return 0, {'pattern_broken': True, 'break_reason': 'Below flagpole start'}
    
    # Age check
    flag_high = flag_data['High'].max()
    days_old = next((i for i in range(1, 11) if data['High'].iloc[-i] == flag_high), 11)
    
    if days_old > PATTERN_AGE_LIMITS['daily']['Bull Flag']:
        return confidence * 0.5, {**pattern_info, 'pattern_stale': True, 'days_old': days_old}
    
    pattern_info['days_since_high'] = days_old
    
    # Technical confirmation
    if macd_line.iloc[-1] > signal_line.iloc[-1]:
        confidence += 15
        pattern_info['macd_bullish'] = True
    
    if histogram.iloc[-1] > histogram.iloc[-3]:
        confidence += 10
        pattern_info['momentum_recovering'] = True
    
    # Volume analysis
    volume_score, volume_info = analyze_volume_pattern(data, "Bull Flag", pattern_info)
    confidence += volume_score
    pattern_info.update(volume_info)
    
    # Near breakout bonus
    if current_price >= flag_high * 0.95:
        confidence += 10
        pattern_info['near_breakout'] = True
    
    # Apply volume confirmation cap
    if not (volume_info.get('good_volume') or volume_info.get('strong_volume') or volume_info.get('exceptional_volume')):
        confidence = min(confidence, MAX_CONFIDENCE_WITHOUT_VOLUME)
        pattern_info['confidence_capped'] = "No volume confirmation"
    
    return confidence, pattern_info

def detect
