'use client';

import Image from 'next/image';
import { useRouter } from 'next/navigation';
import { BlogPost } from '@/types';
import { getImagePath } from '@/lib/utils';

const carouselItems = [
  { image: '/hero-ai-collaboration.jpg', slug: 'feature-engineering-mastery' },
  { image: '/hero-ml-fundamentals.jpg', slug: 'getting-started-with-pytorch' },
  { image: '/hero-data-science.jpg', slug: 'production-ml-best-practices' },
  { image: '/hero-data-scientist.webp', slug: 'deep-learning-optimization' },
  { image: '/icon-data-chart.png', slug: 'nlp-preprocessing-guide' },
  { image: '/bg-tech-abstract.jpg', slug: 'reinforcement-learning-basics' },
  { image: '/screenshot-platform-2023.webp', slug: 'data-visualization-guide' }
];

interface ScrollCarouselProps {
  posts: BlogPost[];
}

export default function ScrollCarousel({ posts }: ScrollCarouselProps) {
  const router = useRouter();
  
  const getPostTitle = (slug: string) => {
    const post = posts.find(p => p.slug === slug);
    return post?.title || 'Blog Post';
  };

  return (
    <div className="carousel-container">
      <div className="carousel-track">
        {[...carouselItems, ...carouselItems].map((item, index) => (
          <div 
            key={`${item.slug}-${index}`} 
            className="carousel-item"
            onClick={() => router.push(`/posts/${item.slug}`)}
          >
            <Image
              src={getImagePath(item.image)}
              alt={`Blog post image ${index + 1}`}
              fill
              className="image-fill"
              sizes="(max-width: 768px) 16rem, 20rem"
              loading="eager"
            />
            <div className="carousel-overlay">
              <h3 className="carousel-title">{getPostTitle(item.slug)}</h3>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}