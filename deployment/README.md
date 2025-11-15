# Deployment Guide - Log Anomaly Detection System

## 📋 Tổng Quan

Hướng dẫn triển khai hệ thống phát hiện bất thường trong log sử dụng Docker và Docker Compose.

## 🚀 Quick Start

### 1. Prerequisites

- Docker >= 20.10
- Docker Compose >= 2.0
- Python 3.9+ (for local development)
- 8GB RAM minimum
- 20GB disk space

### 2. Setup

```bash
# Clone repository (if not already done)
cd code/deployment

# Copy environment variables
cp .env.example .env
# Edit .env with your configuration

# Make scripts executable
chmod +x kafka_setup.sh

# Start infrastructure (Kafka, Zookeeper)
docker-compose up -d zookeeper kafka

# Wait for Kafka to be ready (30 seconds)
sleep 30

# Setup Kafka topics
./kafka_setup.sh

# Start all services
docker-compose up -d
```

### 3. Verify Setup

```bash
# Check all services are running
docker-compose ps

# Check Kafka topics
docker-compose exec kafka kafka-topics --list --bootstrap-server localhost:9092

# Check logs
docker-compose logs -f log-producer
docker-compose logs -f preprocessor
docker-compose logs -f model-service
docker-compose logs -f alert-service
docker-compose logs -f dashboard
```

### 4. Access Dashboard

Open browser: http://localhost:5000

## 📁 Project Structure

```
code/deployment/
├── docker-compose.yml          # Docker Compose configuration
├── Dockerfile                  # Docker image definition
├── requirements.txt            # Python dependencies
├── kafka_setup.sh              # Kafka topics setup script
├── .env.example                # Environment variables example
├── README.md                   # This file
├── src/                        # Application source code
│   ├── log_producer.py         # Log producer service
│   ├── preprocessor.py         # Log preprocessor service
│   ├── model_service.py        # Model inference service
│   ├── alert_service.py        # Alert service
│   └── dashboard/              # Dashboard application
│       ├── app.py              # Flask application
│       ├── templates/          # HTML templates
│       └── static/             # CSS, JS, images
├── config/                     # Configuration files
│   ├── kafka_config.yml        # Kafka configuration
│   ├── model_config.yml        # Model configuration
│   └── alert_config.yml        # Alert configuration
└── models/                     # Trained models
    ├── autoencoder/            # Autoencoder models
    └── logbert/                # LogBERT models
```

## 🔧 Configuration

### Kafka Configuration

Edit `config/kafka_config.yml` to configure Kafka settings.

### Model Configuration

Edit `config/model_config.yml` to configure model paths and parameters.

### Alert Configuration

Edit `config/alert_config.yml` to configure alert channels (Telegram, Email, etc.).

### Environment Variables

Edit `.env` file for environment-specific settings.

## 🧪 Testing

### Test Kafka Producer

```bash
docker-compose exec kafka kafka-console-producer \
  --topic raw-logs \
  --bootstrap-server localhost:9092
```

### Test Kafka Consumer

```bash
docker-compose exec kafka kafka-console-consumer \
  --topic raw-logs \
  --bootstrap-server localhost:9092 \
  --from-beginning
```

### Test Model Service API

```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/predict \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"log": "test log message"}'
```

## 📊 Monitoring

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f model-service
```

### Check Service Health

```bash
# Dashboard health
curl http://localhost:5000/api/health

# Model service health
curl http://localhost:8000/health
```

## 🛠️ Troubleshooting

### Kafka not starting

```bash
# Check Zookeeper is running
docker-compose ps zookeeper

# Check Kafka logs
docker-compose logs kafka

# Restart Kafka
docker-compose restart kafka
```

### Services not connecting to Kafka

```bash
# Verify Kafka is accessible
docker-compose exec kafka kafka-broker-api-versions \
  --bootstrap-server localhost:9092

# Check network
docker network ls
docker network inspect deployment_log-anomaly-network
```

### Model not loading

```bash
# Check model files exist
ls -la models/autoencoder/
ls -la models/logbert/

# Check model service logs
docker-compose logs model-service
```

## 🚀 Production Deployment

For production deployment, see `SME_DEPLOYMENT.md` for detailed instructions.

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Kafka Documentation](https://kafka.apache.org/documentation/)
- [Flask Documentation](https://flask.palletsprojects.com/)

## 📝 Notes

- Default Kafka topics are created automatically
- Models should be placed in `models/` directory before starting services
- Environment variables can be set in `.env` file or docker-compose.yml
- For production, use proper secrets management

