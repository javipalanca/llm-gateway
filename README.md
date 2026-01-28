# LLM Gateway

Intelligent LLM model management system with automatic demand-based scaling.

## 📚 Documentation

All documentation is organized in the [docs/](docs/) folder.

### 📖 Main Documents

- **[docs/README.md](docs/README.md)** - Complete system guide
- **[docs/QUICK_START.md](docs/QUICK_START.md)** - Quick start guide (first 5 minutes)
- **[docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)** - Quick reference of commands
- **[docs/MONITORING.md](docs/MONITORING.md)** - Metrics and dashboards configuration
- **[docs/SECURITY_ENV.md](docs/SECURITY_ENV.md)** - Security configuration

### 🧪 Testing Documentation

- **[docs/TESTS_REFERENCE.md](docs/TESTS_REFERENCE.md)** - Complete tests reference
- **[docs/TESTS_SETUP.md](docs/TESTS_SETUP.md)** - Quick test setup
- **[tests/README.md](tests/README.md)** - Detailed testing guide

See the **[complete documentation index](docs/INDEX.md)** for more details.

## 🚀 Quick Start

```bash
# 1. Clone repository
git clone https://github.com/javipalanca/llm-gateway.git
cd llm-gateway

# 2. Configure environment
cp .env.example .env
# Edit .env and set your passwords and configuration

# 3. Start services
docker-compose up -d

# 4. Check status
curl http://localhost:9010/health
```

## 🔑 Security Configuration

⚠️ **IMPORTANT**: Before using in production:

1. Edit the `.env` file and change default passwords:
   - `POSTGRES_PASSWORD`
   - `LITELLM_MASTER_KEY`
   - `GRAFANA_ADMIN_PASSWORD`

2. Never push the `.env` file to version control (already in `.gitignore`)

3. For usage examples, passwords are read from environment variables

See [docs/SECURITY_ENV.md](docs/SECURITY_ENV.md) for more information.

## 📊 Monitoring

- **Controller**: http://localhost:9010/health
- **LiteLLM**: http://localhost:9001/health
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (user: admin, password: see `.env`)

## 🧪 Tests

```bash
cd tests
./run_tests.sh
```

See [tests/README.md](tests/README.md) for more information.

## 📝 License

MIT License © 2026 Javi Palanca

## 🤝 Contributions

Contributions are welcome. Please open an issue or pull request.
