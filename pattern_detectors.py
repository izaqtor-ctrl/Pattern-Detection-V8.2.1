# pattern_detectors.py
# Pattern Detector V8.1 - Minimal Working Version

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
    
    confidence += 25
    pattern_info['cup_depth'] = str(round(cup_depth * 100, 1)) + "%"
    
    # Handle analysis
    if handle_days > 0:
        handle_low = handle_data['Low'].min()
        current_price = data['Close'].iloc[-1]
        handle_depth = (cup_right - handle_low) / cup_right
        
        if handle_depth > thresholds['max_handle_depth']:
            confidence += 10
            pattern_info['deep_handle'] = str(round(handle_depth * 100, 1)) + "%"
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
    """Simple Inverse Head and Shoulders Detection"""
    confidence = 0
    pattern_info = {}
    
    if len(data) < 30:
        return confidence, pattern_info
    
    # Simple 3-low detection
    lookback = min(60, len(data))
    recent_data = data.tail(lookback)
    
    if len(recent_data) < 20:
        return confidence, pattern_info
    
    # Find lowest point (head)
    head_idx = recent_data['Low'].idxmin()
    head_idx_pos = recent_data.index.get_loc(head_idx)
    head_price = recent_data['Low'].loc[head_idx]
    
    # Need buffer around head
    if head_idx_pos < 5 or head_idx_pos > len(recent_data) - 5:
        return confidence, pattern_info
    
    # Find left shoulder
    left_data = recent_data.iloc[:head_idx_pos]
    if len(left_data) < 5:
        return confidence, pattern_info
    
    left_candidates = []
    for i in range(2, len(left_data) - 2):
        price = left_data['Low'].iloc[i]
        if price <= left_data['Low'].iloc[i-2:i+3].min() and price > head_price:
            left_candidates.append({'idx_pos': i, 'price': price})
    
    if not left_candidates:
        return confidence, pattern_info
    
    left_shoulder = min(left_candidates, key=lambda x: x['price'])
    
    # Find right shoulder
    right_data = recent_data.iloc[head_idx_pos:]
    if len(right_data) < 5:
        return confidence, pattern_info
    
    right_candidates = []
    for i in range(2, len(right_data) - 2):
        price = right_data['Low'].iloc[i]
        if price <= right_data['Low'].iloc[i-2:i+3].min() and price > head_price:
            right_candidates.append({'idx_pos': head_idx_pos + i, 'price': price})
    
    if not right_candidates:
        return confidence, pattern_info
    
    right_shoulder = min(right_candidates, key=lambda x: x['price'])
    
    # Find neckline
    left_neck_data = recent_data.iloc[left_shoulder['idx_pos']:head_idx_pos]
    right_neck_data = recent_data.iloc[head_idx_pos:right_shoulder['idx_pos']]
    
    if len(left_neck_data) < 2 or len(right_neck_data) < 2:
        return confidence, pattern_info
    
    left_neck_price = left_neck_data['High'].max()
    right_neck_price = right_neck_data['High'].max()
    
    # Calculate metrics
    avg_shoulder_price = (left_shoulder['price'] + right_shoulder['price']) / 2
    head_depth = (avg_shoulder_price - head_price) / avg_shoulder_price
    
    pattern_width = right_shoulder['idx_pos'] - left_shoulder['idx_pos']
    
    # Basic validation
    if head_depth < 0.02 or pattern_width < 8 or pattern_width > 100:
        return confidence, pattern_info
    
    # Base confidence
    confidence = 50
    
    # Pattern info
    pattern_info.update({
        'left_shoulder_price': round(left_shoulder['price'], 2),
        'head_price': round(head_price, 2),
        'right_shoulder_price': round(right_shoulder['price'], 2),
        'left_neck_price': round(left_neck_price, 2),
        'right_neck_price': round(right_neck_price, 2),
        'head_depth_percent': str(round(head_depth * 100, 1)) + "%",
        'pattern_width_bars': int(pattern_width)
    })
    
    # Depth scoring
    if head_depth >= 0.05:
        confidence += 15
        if head_depth >= 0.10:
            confidence += 10
            pattern_info['good_head_depth'] = True
    
    # Position relative to neckline
    current_price = data['Close'].iloc[-1]
    neckline_price = (left_neck_price + right_neck_price) / 2
    
    distance_to_neckline = abs(neckline_price - current_price) / current_price
    if distance_to_neckline < 0.05:
        confidence += 15
        pattern_info['near_breakout'] = True
    elif distance_to_neckline < 0.10:
        confidence += 10
        pattern_info['approaching_neckline'] = True
    
    # Technical indicators
    if macd_line.iloc[-1] > signal_line.iloc[-1]:
        confidence += 10
        pattern_info['macd_bullish'] = True
    
    if histogram.iloc[-1] > histogram.iloc[-3]:
        confidence += 10
        pattern_info['momentum_improving'] = True
    
    # Volume analysis
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
        pattern_info['volume_status'] = "Weak Volume (" + str(round(volume_multiplier, 1)) + "x)"
    
    confidence += volume_score
    
    # Apply volume confirmation cap
    if not (pattern_info.get('good_volume') or pattern_info.get('strong_volume') or pattern_info.get('exceptional_volume')):
        confidence = min(confidence, 70)
        pattern_info['confidence_capped'] = "No volume confirmation"
    
    # Age penalty
    bars_since_pattern = len(data) - right_shoulder['idx_pos']
    age_limit = 25 if timeframe == "1wk" else 35
    
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
