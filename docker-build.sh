#!/bin/bash

echo "🐳 Building Docker images..."

# Build the API image
docker build -t funnel-api:latest .

echo "✅ Build complete!"
echo ""
echo "🚀 To run the application:"
echo "   docker-compose up -d"
echo ""
echo "📊 To view logs:"
echo "   docker-compose logs -f api"
echo ""
echo "🛑 To stop:"
echo "   docker-compose down"