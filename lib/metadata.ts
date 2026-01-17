import { Metadata } from 'next';

export const siteConfig = {
  name: 'ML Mondays - Neural Hive',
  description: 'Exploring the frontiers of machine learning, AI research, and neural networks. Weekly insights from the Neural Hive community.',
  url: 'https://proindra.github.io/prajwalindrakh-mlmondays',
  ogImage: '/og-image.png',
};

export function generateMetadata(
  title?: string,
  description?: string,
  image?: string,
  path?: string
): Metadata {
  const metaTitle = title ? `${title} | ${siteConfig.name}` : siteConfig.name;
  const metaDescription = description || siteConfig.description;
  const metaImage = image || siteConfig.ogImage;
  const url = path ? `${siteConfig.url}${path}` : siteConfig.url;

  return {
    metadataBase: new URL(siteConfig.url),
    title: metaTitle,
    description: metaDescription,
    openGraph: {
      title: metaTitle,
      description: metaDescription,
      url,
      siteName: siteConfig.name,
      images: [{ url: metaImage }],
      type: 'website',
    },
    twitter: {
      card: 'summary_large_image',
      title: metaTitle,
      description: metaDescription,
      images: [metaImage],
    },
    robots: {
      index: true,
      follow: true,
    },
  };
}