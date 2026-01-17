import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';
import { marked } from 'marked';

// Configure marked for synchronous parsing
marked.setOptions({
  async: false
});
import { BlogPost } from '@/types';

const postsDirectory = path.join(process.cwd(), 'content/posts');

export function getAllPosts(): BlogPost[] {
  if (!fs.existsSync(postsDirectory)) {
    return [];
  }
  
  const fileNames = fs.readdirSync(postsDirectory);
  const allPostsData = fileNames
    .filter(name => name.endsWith('.md'))
    .map((fileName) => {
      const slug = fileName.replace(/\.md$/, '');
      const fullPath = path.join(postsDirectory, fileName);
      const fileContents = fs.readFileSync(fullPath, 'utf8');
      const matterResult = matter(fileContents);
      
      return {
        id: slug,
        slug,
        title: matterResult.data.title || 'Untitled',
        excerpt: matterResult.data.excerpt || '',
        content: marked(matterResult.content) as string,
        author: matterResult.data.author || 'Neural Hive',
        date: matterResult.data.date || new Date().toISOString(),
        tags: matterResult.data.tags || [],
        readTime: Math.ceil(matterResult.content.split(' ').length / 200),
        image: matterResult.data.image || '/hero-ai-collaboration.jpg'
      } as BlogPost;
    });

  return allPostsData.sort((a, b) => (a.date < b.date ? 1 : -1));
}

export function getPostBySlug(slug: string): BlogPost | null {
  try {
    const fullPath = path.join(postsDirectory, `${slug}.md`);
    const fileContents = fs.readFileSync(fullPath, 'utf8');
    const matterResult = matter(fileContents);
    
    return {
      id: slug,
      slug,
      title: matterResult.data.title || 'Untitled',
      excerpt: matterResult.data.excerpt || '',
      content: marked(matterResult.content) as string,
      author: matterResult.data.author || 'Neural Hive',
      date: matterResult.data.date || new Date().toISOString(),
      tags: matterResult.data.tags || [],
      readTime: Math.ceil(matterResult.content.split(' ').length / 200),
      image: matterResult.data.image || '/hero-ai-collaboration.jpg'
    };
  } catch {
    return null;
  }
}