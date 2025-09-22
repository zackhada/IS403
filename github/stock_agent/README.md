# 🤖 AI Stock Market Agent - LangChain Agentic Framework

A comprehensive agentic framework for tracking AI stock prices, detecting significant market movements, and providing intelligent market analysis. Built with LangChain to demonstrate enterprise-grade agent architecture relevant to companies like Palantir.

## 🎯 Project Overview

This project showcases advanced LangChain concepts and agentic patterns through a real-world financial analysis application. It demonstrates:

- **Agent Architecture**: Multi-tool LangChain agents with reasoning capabilities
- **Real-time Data Processing**: Live stock data ingestion and analysis
- **Advanced Analytics**: Statistical analysis, sentiment analysis, and risk assessment
- **Event-driven Systems**: Real-time alerting and monitoring
- **Interactive Interfaces**: Both CLI and web-based interactions
- **Modular Design**: Clean separation of concerns and extensible architecture

## 🏗️ Architecture

### Core Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Interface │────│  Enhanced Agent  │────│  Stock Data     │
│   (Streamlit)   │    │   (LangChain)    │    │   Manager       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                    ┌───────────┼───────────┐
                    │           │           │
            ┌───────▼──┐ ┌─────▼─────┐ ┌───▼──────┐
            │Analytics │ │Visualization│ │ Alerting │
            │ Engine   │ │   System    │ │  System  │
            └──────────┘ └───────────┘ └──────────┘
```

### Key Technologies

- **LangChain**: Agent framework, tool orchestration, and reasoning
- **OpenAI GPT-4**: Natural language understanding and generation
- **yFinance**: Real-time stock data ingestion
- **Plotly**: Interactive data visualizations
- **Streamlit**: Web interface and dashboard
- **TextBlob**: News sentiment analysis
- **Pandas/NumPy**: Data processing and analysis

## 🚀 Features

### 1. Intelligent Agent System
- **Multi-tool LangChain agent** with advanced reasoning capabilities
- **Natural language queries** for complex market analysis
- **Conversation memory** for context-aware interactions
- **Error handling** and graceful degradation

### 2. Real-time Market Analysis
- **Live stock price tracking** for AI/tech companies
- **Statistical significance detection** using z-scores and volatility analysis
- **Technical indicator calculations** (RSI, Bollinger Bands, Moving Averages)
- **Market correlation analysis** and beta calculations

### 3. Advanced Analytics
- **News sentiment analysis** using NLP techniques
- **Risk assessment** with VaR, drawdown analysis, and volatility modeling
- **Portfolio optimization** with risk-return analysis
- **Pattern recognition** and trend analysis

### 4. Visualization & Dashboards
- **Interactive price charts** with technical indicators
- **Correlation heatmaps** and risk-return scatter plots
- **Portfolio composition** visualizations
- **Real-time dashboard** with key metrics

### 5. Alerting System
- **Real-time monitoring** of price movements and market conditions
- **Customizable alert rules** (price thresholds, volatility spikes, sentiment shifts)
- **Multiple notification channels** (console, email, webhooks)
- **Alert analytics** and historical tracking

## 📦 Installation

### Prerequisites
- Python 3.8+
- OpenAI API key
- Git

### Quick Start

1. **Clone the repository**
```bash
git clone <repository-url>
cd ai-stock-agent
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
# Create .env file
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

4. **Run the basic agent**
```bash
python stock_agent.py
```

5. **Run the enhanced agent**
```bash
python enhanced_stock_agent.py
```

6. **Launch the web interface**
```bash
streamlit run streamlit_app.py
```

## 🎮 Usage

### Command Line Interface

#### Basic Stock Analysis
```bash
python AI_stock_agent.py
```

#### Enhanced Agent with LangChain
```bash
python enhanced_stock_agent.py
```

Example queries:
- "What are the current prices for NVDA, MSFT, and GOOGL?"
- "Show me significant movers in AI stocks today"
- "Analyze sentiment for NVIDIA over the past week"
- "Calculate the beta of PLTR relative to the market"
- "Assess investment risk for a portfolio of NVDA, AMD, and MSFT"

### Web Interface

Launch the Streamlit app:
```bash
streamlit run streamlit_app.py
```

Navigate through:
- **Dashboard**: Market overview and key metrics
- **Agent Chat**: Natural language interaction with the AI agent
- **Stock Analysis**: Detailed technical and fundamental analysis
- **Portfolio Optimizer**: Risk-return analysis and portfolio construction
- **Alert Center**: Real-time monitoring and alert management

### Programmatic Usage

```python
from enhanced_stock_agent import EnhancedStockAgent

# Initialize agent
agent = EnhancedStockAgent()

# Query the agent
response = agent.query("Analyze NVIDIA's recent performance")
print(response)

# Access individual components
stock_data = agent.stock_manager.fetch_stock_data(["NVDA"], 30)
significant_moves = agent.analyzer.detect_significant_moves(stock_data)
```

## 🛠️ Key Components

### 1. Stock Data Manager (`stock_agent.py`)
- Real-time data fetching with caching
- Data normalization and cleaning
- Error handling and retry logic

### 2. Enhanced Agent (`enhanced_stock_agent.py`)
- LangChain agent with custom tools
- Conversation memory and context management
- Advanced reasoning and tool orchestration

### 3. Advanced Analytics (`advanced_analytics.py`)
- News sentiment analysis
- Correlation and beta calculations
- Volatility modeling and risk metrics

### 4. Visualization System (`visualization.py`)
- Interactive Plotly charts
- Dashboard layouts
- Multiple chart types (price, correlation, risk-return)

### 5. Alerting System (`alerting_system.py`)
- Real-time monitoring
- Customizable alert rules
- Multiple notification channels

### 6. Web Interface (`streamlit_app.py`)
- Multi-page dashboard
- Interactive components
- Real-time data updates

## 🔧 Configuration

### Environment Variables
```env
# Required
OPENAI_API_KEY=your_openai_api_key

# Optional
NEWS_API_KEY=your_news_api_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
```

### Alert Configuration
```python
config = {
    'monitor_interval_seconds': 300,
    'notifications': {
        'console': True,
        'email': {
            'enabled': True,
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': 'your_email@gmail.com',
            'password': 'your_app_password'
        }
    }
}
```

## 📊 Monitored Stocks

The system tracks major AI and technology companies:

- **NVDA** - NVIDIA Corporation
- **MSFT** - Microsoft Corporation
- **GOOGL** - Alphabet Inc.
- **AMZN** - Amazon.com Inc.
- **META** - Meta Platforms Inc.
- **AMD** - Advanced Micro Devices
- **AVGO** - Broadcom Inc.
- **PLTR** - Palantir Technologies
- **SNOW** - Snowflake Inc.
- **ORCL** - Oracle Corporation
- **CRM** - Salesforce Inc.

## 🧠 Learning Objectives

This project demonstrates key concepts relevant to companies like Palantir:

### 1. Agent Architecture
- **Tool orchestration** and reasoning
- **Memory management** and context persistence
- **Error handling** and graceful degradation
- **Modular design** and extensibility

### 2. Data Engineering
- **Real-time data ingestion** and processing
- **Data normalization** and cleaning
- **Caching strategies** and performance optimization
- **API integration** and rate limiting

### 3. Advanced Analytics
- **Statistical analysis** and significance testing
- **Machine learning** for pattern recognition
- **Risk modeling** and quantitative finance
- **Natural language processing** for sentiment analysis

### 4. System Design
- **Event-driven architecture** for real-time monitoring
- **Microservices patterns** and separation of concerns
- **Configuration management** and environment handling
- **Scalability** and performance considerations

### 5. User Experience
- **Interactive dashboards** and data visualization
- **Natural language interfaces** for complex queries
- **Real-time updates** and responsive design
- **Multi-modal interaction** (CLI, web, API)

## 🔄 Extending the Framework

### Adding New Data Sources
```python
class CustomDataSource:
    def fetch_data(self, symbols, timeframe):
        # Implement custom data fetching
        pass

# Register with stock manager
stock_manager.add_data_source(CustomDataSource())
```

### Creating Custom Alert Rules
```python
from alerting_system import AlertRule, AlertType

class CustomAlertRule(AlertRule):
    def check_condition(self, data):
        # Implement custom logic
        if custom_condition_met(data):
            return self.create_alert(data)
        return None
```

### Adding New LangChain Tools
```python
from langchain.tools import tool

@tool
def custom_analysis_tool(query: str) -> str:
    """Custom analysis tool description"""
    # Implement tool logic
    return analysis_result

# Add to agent tools
agent.tools.append(custom_analysis_tool)
```

## 🎯 Business Applications

This framework demonstrates capabilities relevant to:

### Financial Services
- **Algorithmic trading** systems
- **Risk management** platforms
- **Portfolio optimization** tools
- **Market research** automation

### Technology Companies
- **Data analytics** platforms
- **Real-time monitoring** systems
- **AI-powered dashboards**
- **Decision support** systems

### Enterprise Software
- **Business intelligence** tools
- **Predictive analytics** platforms
- **Automated reporting** systems
- **Event-driven architectures**

## 🔍 Advanced Features

### Statistical Analysis
- Z-score based significance detection
- Rolling volatility calculations
- Beta and alpha calculations
- Value at Risk (VaR) modeling

### Sentiment Analysis
- News article processing
- TextBlob sentiment scoring
- Confidence level calculations
- Historical sentiment tracking

### Risk Management
- Portfolio risk assessment
- Correlation analysis
- Maximum drawdown calculation
- Diversification metrics

### Technical Indicators
- Simple and exponential moving averages
- Relative Strength Index (RSI)
- Bollinger Bands
- MACD and other oscillators

## 🚀 Production Considerations

### Scalability
- **Horizontal scaling** with multiple agent instances
- **Load balancing** for high-throughput scenarios
- **Caching strategies** for frequently accessed data
- **Database integration** for historical data storage

### Reliability
- **Error handling** and retry mechanisms
- **Health monitoring** and observability
- **Graceful degradation** when external services fail
- **Data validation** and integrity checks

### Security
- **API key management** and rotation
- **Input validation** and sanitization
- **Rate limiting** and abuse prevention
- **Audit logging** for compliance

### Performance
- **Async operations** for concurrent data fetching
- **Memory optimization** for large datasets
- **Query optimization** for complex analytics
- **Response caching** for improved latency

## 📈 Next Steps

1. **Machine Learning Integration**
   - Price prediction models
   - Anomaly detection algorithms
   - Clustering for market regime identification

2. **Additional Data Sources**
   - Options data and volatility surfaces
   - Economic indicators and macro data
   - Social media sentiment tracking

3. **Enhanced Visualizations**
   - 3D volatility surfaces
   - Network graphs for correlation analysis
   - Heat maps for sector performance

4. **Production Deployment**
   - Docker containerization
   - Kubernetes orchestration
   - CI/CD pipeline setup
   - Monitoring and alerting infrastructure

## 🤝 Contributing

This project is designed for learning and demonstration. Key areas for contribution:

- Additional data sources and APIs
- New analytical techniques and indicators
- Enhanced visualization components
- Performance optimizations
- Documentation improvements

## 📄 License

This project is created for educational purposes and career development.

## 📞 Contact

Built to demonstrate advanced LangChain capabilities and agentic frameworks relevant to companies like Palantir.

---

**Built with ❤️ using LangChain, OpenAI, and modern Python technologies**
