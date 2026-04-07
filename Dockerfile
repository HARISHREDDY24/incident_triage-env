FROM python:3.10-slim

WORKDIR /app

# Install uv (the new package manager you just used)
RUN pip install uv

# Copy the lock files first for better caching
COPY pyproject.toml uv.lock ./

# Install the dependencies exactly as locked
RUN uv sync --frozen

# Copy your actual code
COPY . .

# Expose the mandatory Hugging Face port
EXPOSE 7860

# Run the server using the command we defined in pyproject.toml
CMD ["uv", "run", "server"]