FROM oven/bun:1

WORKDIR /app

COPY package.json ./
COPY tsconfig.json ./

RUN bun install

COPY . .

# Build not strictly necessary with bun run, but good practice if we were compiling.
# We will run TS directly.

ENTRYPOINT ["bun", "run", "src/index.ts"]
