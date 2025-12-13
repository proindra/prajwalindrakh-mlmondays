'use client';

import Image from 'next/image';
import { useRouter } from 'next/navigation';
import { BlogPost } from '@/types';

const carouselItems = [
  { image: '/ai-working-group.jpg', slug: 'feature-engineering-mastery' },
  { image: '/AIML.jpg', slug: 'getting-started-with-pytorch' },
  { image: '/data-science22.jpg', slug: 'production-ml-best-practices' },
  { image: '/data-scientist.webp', slug: 'deep-learning-optimization' },
  { image: '/data.png', slug: 'nlp-preprocessing-guide' },
  { image: '/images.jpg', slug: 'reinforcement-learning-basics' },
  { image: '/screenshot-2023.webp', slug: 'data-visualization-guide' }
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
              src={item.image}
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