import { notFound } from 'next/navigation';
import { format } from 'date-fns';
import { Clock, User, ArrowLeft } from 'lucide-react';
import Link from 'next/link';
import Image from 'next/image';
import { getAllPosts, getPostBySlug } from '@/lib/posts';

interface PostPageProps {
  params: Promise<{ slug: string }>;
}

export async function generateStaticParams() {
  const posts = getAllPosts();
  return posts.map((post) => ({
    slug: post.slug,
  }));
}

export default async function PostPage({ params }: PostPageProps) {
  const { slug } = await params;
  const post = getPostBySlug(slug);

  if (!post) {
    notFound();
  }

  const allPosts = getAllPosts();
  const recommendedPosts = allPosts
    .filter(p => p.slug !== post.slug)
    .slice(0, 4);

  return (
    <div className="post-page">
      <Link href="/posts" className="back-link">
        <ArrowLeft size={20} />
        Back to Posts
      </Link>

      <div className="post-layout">
        <div className="post-main">
          {post.image && (
            <div className="post-image">
              <Image
                src={post.image}
                alt={post.title}
                fill
                className="image-fill"
                priority
              />
            </div>
          )}

          <article className="post-article">
            <header className="post-header">
              <h1 className="post-title">{post.title}</h1>
              
              <div className="post-meta">
                <div className="meta-item">
                  <User size={18} />
                  <span>{post.author}</span>
                </div>
                <div className="meta-item">
                  <Clock size={18} />
                  <span>{post.readTime} min read</span>
                </div>
                <span>{format(new Date(post.date), 'MMMM dd, yyyy')}</span>
              </div>

              {post.tags.length > 0 && (
                <div className="post-tags">
                  {post.tags.map((tag) => (
                    <span key={tag} className="post-tag">
                      {tag}
                    </span>
                  ))}
                </div>
              )}
            </header>

            <div 
              className="post-content prose"
              dangerouslySetInnerHTML={{ __html: post.content }}
            />
          </article>

          <div className="post-footer">
            <div className="post-cta">
              <h3>Enjoyed this article?</h3>
              <p>Join our community and contribute your own insights to ML Mondays</p>
              <Link href="/write-for-us" className="btn-primary neural-glow">
                Write for Us
              </Link>
            </div>
          </div>
        </div>

        <div className="post-sidebar">
          <div className="sidebar-content">
            <div className="recommended-posts">
              <h3>Recommended Reading</h3>
              <div className="recommended-list">
                {recommendedPosts.map((recommendedPost) => (
                  <Link
                    key={recommendedPost.slug}
                    href={`/posts/${recommendedPost.slug}`}
                    className="recommended-item"
                  >
                    <div className="recommended-content">
                      {recommendedPost.image && (
                        <div className="recommended-image">
                          <Image
                            src={recommendedPost.image}
                            alt={recommendedPost.title}
                            fill
                            className="image-fill"
                            sizes="64px"
                          />
                        </div>
                      )}
                      <div className="recommended-text">
                        <h4>{recommendedPost.title}</h4>
                        <div className="recommended-meta">
                          <Clock size={12} />
                          <span>{recommendedPost.readTime} min read</span>
                        </div>
                      </div>
                    </div>
                  </Link>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}