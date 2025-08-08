FROM node:18-alpine

WORKDIR /usr/src/app

# Install dependencies
COPY package*.json ./
RUN npm config set fetch-timeout 300000 && \
    npm config set registry https://registry.npmjs.org/ && \
    npm install

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 3000

# Start the development server
CMD ["npm", "start"]