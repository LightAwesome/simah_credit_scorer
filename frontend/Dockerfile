# Build stage
FROM node:18-alpine as build

WORKDIR /usr/src/app

# Install dependencies first (better caching)
COPY package*.json ./
RUN npm config set fetch-timeout 300000 && \
    npm config set registry https://registry.npmjs.org/ && \
    npm install

# Copy the rest of the application
COPY ./ .

# Build the app for production
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built app to nginx html directory
COPY --from=build /usr/src/app/build /usr/share/nginx/html

# Copy nginx configuration if needed (optional)
# COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 3000

CMD ["nginx", "-g", "daemon off;"]