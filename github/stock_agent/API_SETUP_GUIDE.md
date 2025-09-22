# Stock Data API Setup Guide

This guide shows you how to set up reliable stock data APIs that are better alternatives to yfinance.

## 🏆 Recommended APIs (in order of preference)

### 1. Alpha Vantage (Recommended)
- **Free Tier**: 25 requests/day
- **Reliability**: ⭐⭐⭐⭐⭐
- **Setup**: Easy

**Get your free API key:**
1. Go to [alphavantage.co](https://www.alphavantage.co/support/#api-key)
2. Sign up for free account
3. Get your API key instantly

**Usage:**
```bash
export ALPHA_VANTAGE_API_KEY="your_key_here"
```

### 2. Finnhub
- **Free Tier**: 60 calls/minute
- **Reliability**: ⭐⭐⭐⭐⭐
- **Real-time data**: Yes

**Get your free API key:**
1. Go to [finnhub.io](https://finnhub.io/)
2. Sign up for free account
3. Get API key from dashboard

**Usage:**
```bash
export FINNHUB_API_KEY="your_key_here"
```

### 3. IEX Cloud
- **Free Tier**: 500,000 credits/month
- **Reliability**: ⭐⭐⭐⭐⭐
- **Coverage**: Excellent

**Get your free API key:**
1. Go to [iexcloud.io](https://iexcloud.io/)
2. Sign up for free account
3. Get publishable token

**Usage:**
```bash
export IEX_API_KEY="your_key_here"
```

### 4. Twelve Data
- **Free Tier**: 800 requests/day
- **Reliability**: ⭐⭐⭐⭐

**Get your free API key:**
1. Go to [twelvedata.com](https://twelvedata.com/)
2. Sign up for free
3. Get API key

### 5. Polygon.io
- **Free Tier**: 5 calls/minute
- **Reliability**: ⭐⭐⭐⭐⭐
- **Professional grade**: Yes

## 🚀 Quick Setup

### Option 1: Using Environment Variables
```bash
# Add to your ~/.bashrc or ~/.zshrc
export ALPHA_VANTAGE_API_KEY="your_alpha_vantage_key"
export FINNHUB_API_KEY="your_finnhub_key"
export IEX_API_KEY="your_iex_key"
```

### Option 2: Using .env File
Create a `.env` file in your project directory:
```env
# Stock Data API Keys
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
FINNHUB_API_KEY=your_finnhub_key_here
IEX_API_KEY=your_iex_key_here

# OpenAI for LangChain agent
OPENAI_API_KEY=your_openai_key_here
```

## 📝 Code Examples

### Alpha Vantage Example
```python
from alpha_vantage.timeseries import TimeSeries

def fetch_alpha_vantage_data(symbol):
    ts = TimeSeries(key='YOUR_API_KEY', output_format='pandas')
    data, meta_data = ts.get_daily_adjusted(symbol=symbol)
    return data
```

### Finnhub Example
```python
import requests

def fetch_finnhub_data(symbol):
    url = "https://finnhub.io/api/v1/stock/candle"
    params = {
        'symbol': symbol,
        'resolution': 'D',
        'from': start_timestamp,
        'to': end_timestamp,
        'token': 'YOUR_API_KEY'
    }
    response = requests.get(url, params=params)
    return response.json()
```

### IEX Cloud Example
```python
import requests

def fetch_iex_data(symbol):
    url = f"https://cloud.iexapis.com/stable/stock/{symbol}/chart/1m"
    params = {'token': 'YOUR_API_KEY'}
    response = requests.get(url, params=params)
    return response.json()
```

## 🛡️ API Rate Limits & Best Practices

### Alpha Vantage
- **Limit**: 25 requests/day (free)
- **Best Practice**: Cache data for 4+ hours
- **Error Handling**: Watch for "API call frequency" messages

### Finnhub
- **Limit**: 60 calls/minute (free)
- **Best Practice**: Batch requests when possible
- **Error Handling**: Watch for 429 status codes

### IEX Cloud
- **Limit**: Based on credits (500k/month free)
- **Best Practice**: Use efficient endpoints
- **Error Handling**: Monitor credit usage

## 🔧 Integration with Our Stock Agent

The `multi_source_stock_agent.py` automatically tries these APIs in order:

1. **Alpha Vantage** (if API key available)
2. **Finnhub** (if API key available)
3. **IEX Cloud** (if API key available)
4. **yfinance** (fallback)
5. **Sample Data** (demo mode)

## 🆚 Comparison with yfinance

| Feature | yfinance | Alpha Vantage | Finnhub | IEX Cloud |
|---------|----------|---------------|---------|-----------|
| **Reliability** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Rate Limits** | Unstable | 25/day | 60/min | 500k/month |
| **Real-time** | No | No | Yes | Yes |
| **Free Tier** | Yes | Yes | Yes | Yes |
| **API Key Required** | No | Yes | Yes | Yes |
| **Professional Support** | No | Yes | Yes | Yes |

## 💡 Pro Tips

1. **Use Multiple APIs**: Set up 2-3 APIs for redundancy
2. **Cache Data**: Avoid hitting rate limits by caching responses
3. **Monitor Usage**: Track your API usage to avoid limits
4. **Error Handling**: Always have fallback strategies
5. **Production**: Consider paid tiers for production apps

## 🚨 Common Issues & Solutions

### Issue: "API call frequency exceeded"
**Solution**: Implement exponential backoff or upgrade to paid tier

### Issue: "Invalid API key"
**Solution**: Double-check key format and environment variable setup

### Issue: "No data returned"
**Solution**: Check symbol format (some APIs use different formats)

### Issue: Rate limiting
**Solution**: Implement request queuing and delays between calls

## 🎯 Recommended Setup for Learning

For your LangChain learning project:

1. **Start with Alpha Vantage** (most reliable, good free tier)
2. **Add Finnhub** as backup (good real-time data)
3. **Keep yfinance** as final fallback
4. **Use sample data** for development/testing

This gives you a robust, production-ready data pipeline that companies like Palantir would appreciate!
