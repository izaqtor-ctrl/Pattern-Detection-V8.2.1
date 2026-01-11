# pattern_detectors.py
# Pattern Detector V8.2.2 - ChatGPT-Aligned IHS with Proper Pivot Detection

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
        volume_info['volume_status'] = "Exceptional Volume (" + str(round(volume_multiplier, 1)) + "x)"
    elif volume_multiplier >= VOLUME_THRESHOLDS['strong']:
        volume_score += VOLUME_SCORE_POINTS['strong']
        volume_info['strong_volume'] = True
        volume_info['volume_status'] = "Strong Volume (" + str(round(volume_multiplier, 1)) + "x)"
    elif volume_multiplier >= VOLUME_THRESHOLDS['good']:
        volume_score += VOLUME_SCORE_POINTS['good']
        volume_info['good_volume'] = True
        volume_info['volume_status'] = "Good Volume (" + str(round(volume_multiplier, 1)) + "x)"
    else:
        volume_info['weak_volume'] = True
        volume_info['volume_status'] = "Weak Volume (" + str(round(volume_multiplier, 1)) + "x)"
    
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

def detect_inside_bar(data, macd_line, signal_line, histogram, market_context, timeframe="daily"):
    """Detect Inside Bar pattern"""
    confidence = 0
    pattern_info = {}
    
    if len(data) < 5:
        return confidence, pattern_info
    
    # Adjust lookback based on timeframe
    if timeframe == "1wk":
        max_lookback_range = range(-1, -7, -1)
        aging_threshold = -8
        pattern_info['timeframe'] = 'Weekly'
    else:
        max_lookback_range = range(-1, -5, -1)
        aging_threshold = -6
        pattern_info['timeframe'] = 'Daily'
    
    # Look for inside bar pattern
    mother_bar_idx = None
    inside_bars_count = 0
    inside_bar_indices = []
    
    for i in max_lookback_range:
        try:
            current_bar = data.iloc[i]
            previous_bar = data.iloc[i-1]
            
            is_inside = (current_bar['High'] <= previous_bar['High'] and 
                        current_bar['Low'] >= previous_bar['Low'] and
                        current_bar['High'] < previous_bar['High'] and
                        current_bar['Low'] > previous_bar['Low'])
            
            mother_is_green = previous_bar['Close'] > previous_bar['Open']
            inside_is_red = current_bar['Close'] < current_bar['Open']
            
            if is_inside and mother_is_green and inside_is_red:
                if inside_bars_count == 0:
                    mother_bar_idx = i - 1
                    inside_bar_indices.append(i)
                    inside_bars_count = 1
                elif inside_bars_count == 1 and i == inside_bar_indices[0] - 1:
                    inside_bar_indices.append(i)
                    inside_bars_count = 2
                    break
                else:
                    break
            else:
                break
        except (IndexError, KeyError):
            break
    
    if inside_bars_count == 0:
        return confidence, pattern_info
    
    mother_bar = data.iloc[mother_bar_idx]
    latest_inside_bar = data.iloc[inside_bar_indices[0]]
    
    mother_is_green = mother_bar['Close'] > mother_bar['Open']
    inside_is_red = latest_inside_bar['Close'] < latest_inside_bar['Open']
    
    if not (mother_is_green and inside_is_red):
        return confidence, pattern_info
    
    base_confidence = 35 if timeframe == "1wk" else 30
    confidence += base_confidence
    
    pattern_info['mother_bar_high'] = mother_bar['High']
    pattern_info['mother_bar_low'] = mother_bar['Low']
    pattern_info['inside_bar_high'] = latest_inside_bar['High']
    pattern_info['inside_bar_low'] = latest_inside_bar['Low']
    pattern_info['inside_bars_count'] = inside_bars_count
    pattern_info['color_validated'] = True
    pattern_info['mother_bar_color'] = 'Green'
    pattern_info['inside_bar_color'] = 'Red'
    
    confidence += 15
    pattern_info['proper_color_combo'] = True
    
    if inside_bars_count == 1:
        confidence += 15
        pattern_info['single_inside_bar'] = True
    else:
        confidence += 10
        pattern_info['double_inside_bar'] = True
    
    mother_bar_range = mother_bar['High'] - mother_bar['Low']
    inside_bar_range = latest_inside_bar['High'] - latest_inside_bar['Low']
    
    if mother_bar_range > 0:
        size_ratio = inside_bar_range / mother_bar_range
        pattern_info['size_ratio'] = str(round(size_ratio * 100, 1)) + "%"
        
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
    
    # Technical confirmation
    if macd_line.iloc[-1] > signal_line.iloc[-1]:
        confidence += 15
        pattern_info['macd_bullish'] = True
    
    if histogram.iloc[-1] > histogram.iloc[-3]:
        confidence += 10
        pattern_info['momentum_improving'] = True
    
    current_price = data['Close'].iloc[-1]
    if current_price >= latest_inside_bar['Low'] * 0.98:
        confidence += 10
        pattern_info['price_in_range'] = True
    
    volume_score, volume_info = analyze_volume_pattern(data, "Inside Bar", pattern_info)
    confidence += volume_score
    pattern_info.update(volume_info)
    
    if not (volume_info.get('good_volume') or volume_info.get('strong_volume') or volume_info.get('exceptional_volume')):
        confidence = min(confidence, MAX_CONFIDENCE_WITHOUT_VOLUME)
        pattern_info['confidence_capped'] = "No volume confirmation"
    
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
    pattern_info['initial_ascension'] = str(round(initial_gain * 100, 1)) + "%"
    
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
        confidence = confidence * 0.5
        pattern_info['pattern_stale'] = True
        pattern_info['days_old'] = days_old
        return confidence, pattern_info
    
    if current_price < descent_low * 0.95:
        return 0, {'pattern_broken': True, 'break_reason': 'Below support'}
    
    if macd_line.iloc[-1] > signal_line.iloc[-1]:
        confidence += 10
        pattern_info['macd_bullish'] = True
    
    volume_score, volume_info = analyze_volume_pattern(data, "Flat Top Breakout", pattern_info)
    confidence += volume_score
    pattern_info.update(volume_info)
    
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
    pattern_info['flagpole_gain'] = str(round(flagpole_gain * 100, 1)) + "%"
    
    flag_data = data.tail(15)
    flag_start = data['Close'].iloc[-flagpole_end]
    current_price = data['Close'].iloc[-1]
    
    pullback = (current_price - flag_start) / flag_start
    pullback_range = thresholds['pullback_range']
    if pullback_range[0] <= pullback <= pullback_range[1]:
        confidence += 20
        pattern_info['flag_pullback'] = str(round(pullback * 100, 1)) + "%"
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
        confidence = confidence * 0.5
        pattern_info['pattern_stale'] = True
        pattern_info['days_old'] = days_old
        return confidence, pattern_info
    
    pattern_info['days_since_high'] = days_old
    
    if macd_line.iloc[-1] > signal_line.iloc[-1]:
        confidence += 15
        pattern_info['macd_bullish'] = True
    
    if histogram.iloc[-1] > histogram.iloc[-3]:
        confidence += 10
        pattern_info['momentum_recovering'] = True
    
    volume_score, volume_info = analyze_volume_pattern(data, "Bull Flag", pattern_info)
    confidence += volume_score
    pattern_info.update(volume_info)
    
    if current_price >= flag_high * 0.95:
        confidence += 10
        pattern_info['near_breakout'] = True
    
    if not (volume_info.get('good_volume') or volume_info.get('strong_volume') or volume_info.get('exceptional_volume')):
        confidence = min(confidence, MAX_CONFIDENCE_WITHOUT_VOLUME)
        pattern_info['confidence_capped'] = "No volume confirmation"
    
    return confidence, pattern_info

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
    
    # Base confidence for valid cup
    confidence += 25
    pattern_info['cup_depth'] = str(round(cup_depth * 100, 1)) + "%"
    
    # Bonus for ideal cup depth (10-18% is sweet spot for swing trading)
    ideal_min = thresholds.get('ideal_cup_depth', 0.12) - 0.03
    ideal_max = thresholds.get('ideal_cup_depth', 0.15) + 0.03
    if ideal_min <= cup_depth <= ideal_max:
        confidence += 15
        pattern_info['ideal_cup_depth'] = True
    elif cup_depth > thresholds['max_cup_depth'] * 0.8:  # Very deep (>24%)
        confidence *= 0.85  # Slight penalty - takes longer to recover
        pattern_info['deep_cup'] = True
    
    # Handle analysis
    if handle_days > 0:
        handle_low = handle_data['Low'].min()
        current_price = data['Close'].iloc[-1]
        handle_depth = (cup_right - handle_low) / cup_right
        
        if handle_depth > thresholds['max_handle_depth']:
            confidence *= 0.6  # FIXED: Penalize deep handles (was incorrectly rewarding)
            pattern_info['handle_too_deep'] = str(round(handle_depth * 100, 1)) + "% (exceeds max)"
        elif handle_depth <= 0.08:
            confidence += 20
            pattern_info['perfect_handle'] = str(round(handle_depth * 100, 1)) + "%"
        elif handle_depth <= 0.15:
            confidence += 15
            pattern_info['good_handle'] = str(round(handle_depth * 100, 1)) + "%"
        else:
            confidence += 10
            pattern_info['acceptable_handle'] = str(round(handle_depth * 100, 1)) + "%"
        
        if handle_days > 25:
            confidence *= 0.8
            pattern_info['long_handle'] = str(handle_days) + " days"
        elif handle_days <= 10:
            confidence += 10
            pattern_info['short_handle'] = str(handle_days) + " days"
        elif handle_days <= 20:
            confidence += 5
            pattern_info['medium_handle'] = str(handle_days) + " days"
    else:
        confidence += 10
        pattern_info['forming_handle'] = "Handle forming"
    
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
    
    if macd_line.iloc[-1] > signal_line.iloc[-1]:
        confidence += 10
        pattern_info['macd_bullish'] = True
    
    volume_score, volume_info = analyze_volume_pattern(data, "Cup Handle", pattern_info)
    confidence += volume_score
    pattern_info.update(volume_info)
    
    if confidence < 35:
        return confidence, pattern_info
    
    if not (volume_info.get('good_volume') or volume_info.get('strong_volume') or volume_info.get('exceptional_volume')):
        confidence = min(confidence, MAX_CONFIDENCE_WITHOUT_VOLUME)
        pattern_info['confidence_capped'] = "No volume confirmation"
    
    return confidence, pattern_info

def detect_inverse_head_shoulders(data, macd_line, signal_line, histogram, market_context, timeframe="daily"):
    """
    ChatGPT-Aligned Inverse Head & Shoulders Detection
    Implements minimal, swing-tradable ruleset with proper pivot validation
    
    Key improvements over V8.2.1:
    - Proper N-left/N-right pivot detection (not just local mins)
    - Neckline slope validation
    - Breakout confirmation (must be above neckline)
    - Freshness filter (only recent breakouts)
    - Tighter width constraints (12-35 days)
    """
    confidence = 0
    pattern_info = {}
    
    if len(data) < 30:
        return confidence, pattern_info
    
    thresholds = PATTERN_THRESHOLDS["Inverse Head Shoulders"]
    
    # Determine pivot strength based on timeframe
    N = thresholds.get('pivot_strength_daily', 3) if timeframe != "1wk" else thresholds.get('pivot_strength_weekly', 2)
    
    # Lookback window
    lookback = min(60, len(data))
    recent_data = data.tail(lookback).reset_index(drop=True)
    
    if len(recent_data) < 20:
        return confidence, pattern_info
    
    # ========================================
    # STEP 1: Find Pivot Lows (ChatGPT Method)
    # ========================================
    def is_pivot_low(idx, data, N):
        """
        Check if idx is a pivot low with N bars left and right
        ChatGPT: "A bar is a pivot low if its low is the lowest among N bars left and N bars right"
        """
        if idx < N or idx >= len(data) - N:
            return False
        
        pivot_low = data['Low'].iloc[idx]
        
        # Check N bars to the left
        for i in range(idx - N, idx):
            if data['Low'].iloc[i] <= pivot_low:
                return False
        
        # Check N bars to the right
        for i in range(idx + 1, idx + N + 1):
            if data['Low'].iloc[i] <= pivot_low:
                return False
        
        return True
    
    # Find all pivot lows
    pivot_lows = []
    for i in range(N, len(recent_data) - N):
        if is_pivot_low(i, recent_data, N):
            pivot_lows.append({
                'idx': i,
                'price': recent_data['Low'].iloc[i]
            })
    
    if len(pivot_lows) < 3:
        return confidence, pattern_info
    
    # ========================================
    # STEP 2: Identify Head (lowest pivot)
    # ========================================
    head = min(pivot_lows, key=lambda x: x['price'])
    head_idx = head['idx']
    head_price = head['price']
    
    # Need pivots on both sides of head
    left_pivots = [p for p in pivot_lows if p['idx'] < head_idx]
    right_pivots = [p for p in pivot_lows if p['idx'] > head_idx]
    
    if not left_pivots or not right_pivots:
        return confidence, pattern_info
    
    # ========================================
    # STEP 3: Find Left and Right Shoulders
    # ========================================
    # ChatGPT: Shoulders should be higher than head
    # Find the pivot closest to head on each side
    left_shoulder = max(left_pivots, key=lambda x: x['idx'])  # Closest to head on left
    right_shoulder = min(right_pivots, key=lambda x: x['idx'])  # Closest to head on right
    
    ls_price = left_shoulder['price']
    rs_price = right_shoulder['price']
    ls_idx = left_shoulder['idx']
    rs_idx = right_shoulder['idx']
    
    # ========================================
    # STEP 4: Validate Head Depth
    # ========================================
    # ChatGPT: Head must be ≥4-6% below shoulders
    avg_shoulder_price = (ls_price + rs_price) / 2
    head_depth_pct = (avg_shoulder_price - head_price) / avg_shoulder_price
    
    min_head_depth = thresholds.get('min_head_depth', 0.04)
    max_head_depth = thresholds.get('max_head_depth', 0.40)
    
    if head_depth_pct < min_head_depth or head_depth_pct > max_head_depth:
        return confidence, pattern_info
    
    # ========================================
    # STEP 5: Validate Shoulder Symmetry
    # ========================================
    # ChatGPT: Shoulders should be within ≤6-8% of each other
    shoulder_diff_pct = abs(ls_price - rs_price) / avg_shoulder_price
    max_shoulder_diff = thresholds.get('max_shoulder_diff_pct', 0.08)
    
    if shoulder_diff_pct > max_shoulder_diff:
        return confidence, pattern_info
    
    # Calculate symmetry score (for bonuses)
    symmetry_score = 1 - shoulder_diff_pct
    
    # ========================================
    # STEP 6: Validate Pattern Width
    # ========================================
    # ChatGPT: 12-35 days for daily, 8-20 weeks for weekly
    pattern_width = rs_idx - ls_idx
    
    min_width = thresholds.get('min_pattern_width_daily', 12) if timeframe != "1wk" else thresholds.get('min_pattern_width_weekly', 8)
    max_width = thresholds.get('max_pattern_width_daily', 35) if timeframe != "1wk" else thresholds.get('max_pattern_width_weekly', 20)
    
    if pattern_width < min_width or pattern_width > max_width:
        return confidence, pattern_info
    
    # ========================================
    # STEP 7: Find and Validate Neckline
    # ========================================
    # ChatGPT: Find peak between LS-Head and Head-RS
    left_section = recent_data.iloc[ls_idx:head_idx+1]
    right_section = recent_data.iloc[head_idx:rs_idx+1]
    
    if len(left_section) < 2 or len(right_section) < 2:
        return confidence, pattern_info
    
    # Find neckline peaks
    left_peak_idx = left_section['High'].idxmax()
    right_peak_idx = right_section['High'].idxmax()
    
    left_peak_price = left_section['High'].loc[left_peak_idx]
    right_peak_price = right_section['High'].loc[right_peak_idx]
    
    # ChatGPT: Peaks should be at least 2-3 bars away from troughs
    min_separation = thresholds.get('neckline_min_separation', 2)
    left_peak_pos = left_section.index.get_loc(left_peak_idx)
    right_peak_pos = right_section.index.get_loc(right_peak_idx)
    
    if left_peak_pos < min_separation or right_peak_pos < min_separation:
        return confidence, pattern_info
    
    # ChatGPT: Validate neckline slope (≤6-8% variance)
    neckline_slope = abs(left_peak_price - right_peak_price) / left_peak_price
    max_slope = thresholds.get('neckline_slope_max', 0.08)
    
    if neckline_slope > max_slope:
        confidence *= 0.7  # Penalty for steep neckline
        pattern_info['steep_neckline'] = str(round(neckline_slope * 100, 1)) + "%"
    
    neckline_price = (left_peak_price + right_peak_price) / 2
    
    # ========================================
    # STEP 8: Breakout Confirmation
    # ========================================
    # ChatGPT: Must close ≥0.5% above neckline
    current_price = data['Close'].iloc[-1]
    breakout_buffer = thresholds.get('breakout_buffer', 0.005)  # 0.5%
    
    has_broken_out = current_price >= neckline_price * (1 + breakout_buffer)
    
    if not has_broken_out:
        # Pattern exists but hasn't broken out yet
        confidence = 40  # Lower confidence for incomplete patterns
        pattern_info['awaiting_breakout'] = True
        pattern_info['neckline_distance'] = str(round((neckline_price - current_price) / current_price * 100, 1)) + "%"
    else:
        # Pattern has broken out
        confidence = 65  # Higher base for confirmed breakouts
        pattern_info['breakout_confirmed'] = True
        
        # ========================================
        # STEP 9: Freshness Filter
        # ========================================
        # ChatGPT: Breakout must be within last ≤5 bars (daily) or ≤3 bars (weekly)
        # Find the breakout bar
        breakout_bar = None
        for i in range(len(data) - 1, max(0, len(data) - 10), -1):
            if data['Close'].iloc[i] >= neckline_price * (1 + breakout_buffer):
                breakout_bar = len(data) - 1 - i
                break
        
        freshness_limit = thresholds.get('freshness_bars_daily', 5) if timeframe != "1wk" else thresholds.get('freshness_bars_weekly', 3)
        
        if breakout_bar is not None and breakout_bar > freshness_limit:
            confidence *= 0.5  # Major penalty for stale breakouts
            pattern_info['stale_breakout'] = str(breakout_bar) + " bars ago"
        elif breakout_bar is not None and breakout_bar <= freshness_limit:
            confidence += 15  # Bonus for fresh breakout
            pattern_info['fresh_breakout'] = str(breakout_bar) + " bars ago"
    
    # ========================================
    # STEP 10: Volume Confirmation
    # ========================================
    # ChatGPT: Breakout volume ≥1.3x average (required for daily)
    avg_volume = data['Volume'].tail(20).mean()
    current_volume = data['Volume'].iloc[-1]
    volume_multiplier = current_volume / avg_volume
    
    volume_score = 0
    if volume_multiplier >= 2.0:
        volume_score += 25
        pattern_info['exceptional_volume'] = True
        pattern_info['volume_status'] = "Exceptional Volume (" + str(round(volume_multiplier, 1)) + "x)"
    elif volume_multiplier >= 1.5:
        volume_score += 20
        pattern_info['strong_volume'] = True
        pattern_info['volume_status'] = "Strong Volume (" + str(round(volume_multiplier, 1)) + "x)"
    elif volume_multiplier >= 1.3:
        volume_score += 15
        pattern_info['good_volume'] = True
        pattern_info['volume_status'] = "Good Volume (" + str(round(volume_multiplier, 1)) + "x)"
    else:
        pattern_info['weak_volume'] = True
        pattern_info['volume_status'] = "Weak Volume (" + str(round(volume_multiplier, 1)) + "x)"
    
    confidence += volume_score
    
    # ChatGPT: Volume confirmation required for daily (can be optional for 4H)
    if timeframe != "1wk" and volume_multiplier < thresholds.get('breakout_volume_min', 1.3):
        confidence = min(confidence, MAX_CONFIDENCE_WITHOUT_VOLUME)
        pattern_info['confidence_capped'] = "Volume below 1.3x requirement"
    
    # ========================================
    # STEP 11: Scoring Bonuses/Penalties
    # ========================================
    
    # Pattern info
    pattern_info.update({
        'left_shoulder_price': round(ls_price, 2),
        'head_price': round(head_price, 2),
        'right_shoulder_price': round(rs_price, 2),
        'left_neck_price': round(left_peak_price, 2),
        'right_neck_price': round(right_peak_price, 2),
        'neckline_price': round(neckline_price, 2),
        'head_depth_percent': str(round(head_depth_pct * 100, 1)) + "%",
        'pattern_width_bars': int(pattern_width),
        'shoulder_symmetry': str(round(symmetry_score * 100, 1)) + "%",
        'neckline_slope': str(round(neckline_slope * 100, 1)) + "%"
    })
    
    # ChatGPT Scoring: Start at 60, adjust
    
    # Head depth bonuses (ideal 5-20%)
    ideal_min = thresholds.get('ideal_head_depth_min', 0.05)
    ideal_max = thresholds.get('ideal_head_depth_max', 0.20)
    
    if ideal_min <= head_depth_pct <= ideal_max:
        confidence += 15
        pattern_info['ideal_head_depth'] = True
    elif head_depth_pct < ideal_min:
        confidence += 5
        pattern_info['shallow_head'] = True
    
    # Shoulder symmetry bonuses (ChatGPT: ≤5% daily = tight)
    excellent_sym = thresholds.get('excellent_symmetry', 0.70)
    good_sym = thresholds.get('good_symmetry', 0.50)
    
    if symmetry_score >= excellent_sym:
        confidence += 15
        pattern_info['excellent_symmetry'] = True
    elif symmetry_score >= good_sym:
        confidence += 10
        pattern_info['good_symmetry'] = True
    elif shoulder_diff_pct <= 0.05:  # ≤5% difference (very tight)
        confidence += 10
        pattern_info['tight_shoulders'] = str(round(shoulder_diff_pct * 100, 1)) + "%"
    
    # Neckline slope bonuses
    if neckline_slope <= 0.03:  # ≤3% slope
        confidence += 5
        pattern_info['flat_neckline'] = True
    
    # RS higher than LS (bullish)
    if rs_price > ls_price:
        confidence += 5
        pattern_info['ascending_shoulders'] = True
    
    # Pattern width bonuses/penalties
    compact_threshold = thresholds.get('compact_pattern_days', 20)
    extended_threshold = thresholds.get('extended_pattern_days', 30)
    
    if pattern_width <= compact_threshold:
        confidence += 10
        pattern_info['compact_pattern'] = True
    elif pattern_width >= extended_threshold:
        confidence *= 0.85
        pattern_info['extended_pattern'] = True
    
    # Technical indicators
    if macd_line.iloc[-1] > signal_line.iloc[-1]:
        confidence += 8
        pattern_info['macd_bullish'] = True
    
    if histogram.iloc[-1] > histogram.iloc[-3]:
        confidence += 8
        pattern_info['momentum_improving'] = True
    
    # Apply volume confirmation cap
    if not (pattern_info.get('good_volume') or pattern_info.get('strong_volume') or pattern_info.get('exceptional_volume')):
        confidence = min(confidence, MAX_CONFIDENCE_WITHOUT_VOLUME)
        if 'confidence_capped' not in pattern_info:
            pattern_info['confidence_capped'] = "No volume confirmation"
    
    # ========================================
    # STEP 12: Age Penalty
    # ========================================
    # How long ago did the pattern complete?
    bars_since_rs = len(data) - 1 - (len(data) - len(recent_data) + rs_idx)
    
    timeframe_key = "weekly" if timeframe == "1wk" else "daily"
    age_limit = PATTERN_AGE_LIMITS.get(timeframe_key, {}).get("Inverse Head Shoulders", 20)
    
    if bars_since_rs > age_limit:
        confidence *= 0.6
        pattern_info['pattern_too_old'] = str(bars_since_rs) + " bars ago"
    
    # ========================================
    # STEP 13: Invalidation Check
    # ========================================
    # Price should not be below head (major invalidation)
    if current_price < head_price * 0.98:
        confidence *= 0.4
        pattern_info['below_head_warning'] = "Price near/below head - pattern compromised"
    
    return confidence, pattern_info
    current_volume = data['Volume'].iloc[-1]
    volume_multiplier = current_volume / avg_volume
    
    volume_score = 0
    if volume_multiplier >= 2.0:
        volume_score += 25
        pattern_info['exceptional_volume'] = True
        pattern_info['volume_status'] = "Exceptional Volume (" + str(round(volume_multiplier, 1)) + "x)"
    elif volume_multiplier >= 1.5:
        volume_score += 20
        pattern_info['strong_volume'] = True
        pattern_info['volume_status'] = "Strong Volume (" + str(round(volume_multiplier, 1)) + "x)"
    elif volume_multiplier >= 1.3:
        volume_score += 15
        pattern_info['good_volume'] = True
        pattern_info['volume_status'] = "Good Volume (" + str(round(volume_multiplier, 1)) + "x)"
    else:
        pattern_info['volume_status'] = "Weak Volume (" + str(round(volume_multiplier, 1)) + "x)"
    
    confidence += volume_score
    
    # Apply volume confirmation cap
    if not (pattern_info.get('good_volume') or pattern_info.get('strong_volume') or pattern_info.get('exceptional_volume')):
        confidence = min(confidence, 70)
        pattern_info['confidence_capped'] = "No volume confirmation"
    
    # Symmetry scoring (if we can calculate it)
    try:
        shoulder_price_diff = abs(left_shoulder['price'] - right_shoulder['price'])
        avg_shoulder_price = (left_shoulder['price'] + right_shoulder['price']) / 2
        symmetry_score = 1 - (shoulder_price_diff / avg_shoulder_price)
        
        excellent_sym = thresholds.get('excellent_symmetry', 0.70)
        good_sym = thresholds.get('good_symmetry', 0.50)
        min_sym = thresholds.get('min_symmetry', 0.40)
        
        if symmetry_score >= excellent_sym:
            confidence += 15
            pattern_info['excellent_symmetry'] = str(round(symmetry_score * 100, 1)) + "%"
        elif symmetry_score >= good_sym:
            confidence += 10
            pattern_info['good_symmetry'] = str(round(symmetry_score * 100, 1)) + "%"
        elif symmetry_score >= min_sym:
            confidence += 5
            pattern_info['acceptable_symmetry'] = str(round(symmetry_score * 100, 1)) + "%"
        else:
            confidence *= 0.85  # Penalty for poor symmetry
            pattern_info['poor_symmetry'] = str(round(symmetry_score * 100, 1)) + "%"
    except:
        pass  # If symmetry calculation fails, skip it
    
    # Age penalty using config
    bars_since_pattern = len(data) - right_shoulder['idx_pos']
    timeframe_key = "weekly" if timeframe == "1wk" else "daily"
    age_limit = PATTERN_AGE_LIMITS.get(timeframe_key, {}).get("Inverse Head Shoulders", 35)
    
    if bars_since_pattern > age_limit:
        confidence *= 0.8
        pattern_info['pattern_aging'] = True
        pattern_info['age_bars'] = int(bars_since_pattern)
    
    # Invalidation check
    if current_price < head_price * 0.97:
        confidence *= 0.6
        pattern_info['below_head_warning'] = "Price near/below head level"
    
    return confidence, pattern_info

def detect_pattern(data, pattern_type, market_context, timeframe="daily"):
    """Main pattern detection function"""
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
    
    return confidence >= 55, confidence, pattern_info
