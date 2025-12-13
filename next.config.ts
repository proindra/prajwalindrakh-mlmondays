import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: 'export',
  trailingSlash: true,
  basePath: '/prajwalindrakh-mlmondays',
  images: {
    unoptimized: true
  }
};

export default nextConfig;
