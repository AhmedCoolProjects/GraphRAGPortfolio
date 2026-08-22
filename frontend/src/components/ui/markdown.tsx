"use client";

import { cn } from "@/lib/utils";
import ReactMarkdown, { Components } from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import { useState, useCallback, useMemo } from "react";
import { Lightbox } from "./lightbox";
import { ZoomIn } from "lucide-react";

interface MarkdownProps {
  content: string;
  className?: string;
  images?: { image: string; alt: string }[];
}

export function Markdown({ content, className, images: externalImages }: MarkdownProps) {
  const [lightboxOpen, setLightboxOpen] = useState(false);
  const [lightboxIndex, setLightboxIndex] = useState(0);

  // Combine inline images from content with external images
  const allImages = useMemo(() => {
    const inline: { image: string; alt: string }[] = [];
    const imageRegex = /!\[(.*?)\]\s*\((.*?)\)/g;
    let match;
    
    while ((match = imageRegex.exec(content)) !== null) {
      inline.push({ alt: match[1], image: match[2] });
    }
    
    return [...inline, ...(externalImages || [])];
  }, [content, externalImages]);

  const openLightbox = useCallback((index: number) => {
    setLightboxIndex(index);
    setLightboxOpen(true);
  }, []);

  const cleanContent = useMemo(() => {
    // Remove markdown image syntax since we handle them separately
    return content.replace(/!\[(.*?)\]\s*\((.*?)\)/g, "").trim();
  }, [content]);

  const components: Components = {
    h1: ({ className, ...props }) => (
      <h1
        className={cn(
          "text-2xl font-bold mt-6 mb-3 text-neutral-900 dark:text-neutral-100",
          className
        )}
        {...props}
      />
    ),
    h2: ({ className, ...props }) => (
      <h2
        className={cn(
          "text-xl font-bold mt-5 mb-3 text-neutral-900 dark:text-neutral-100",
          className
        )}
        {...props}
      />
    ),
    h3: ({ className, ...props }) => (
      <h3
        className={cn(
          "text-lg font-semibold mt-4 mb-2 text-neutral-900 dark:text-neutral-100",
          className
        )}
        {...props}
      />
    ),
    p: ({ className, ...props }) => (
      <p
        className={cn("mb-3 text-neutral-700 dark:text-neutral-300", className)}
        {...props}
      />
    ),
    ul: ({ className, ...props }) => (
      <ul
        className={cn("list-disc list-outside ml-5 my-2 space-y-1 text-neutral-700 dark:text-neutral-300", className)}
        {...props}
      />
    ),
    ol: ({ className, ...props }) => (
      <ol
        className={cn("list-decimal list-outside ml-5 my-2 space-y-1 text-neutral-700 dark:text-neutral-300", className)}
        {...props}
      />
    ),
    li: ({ className, ...props }) => (
      <li className={cn("", className)} {...props} />
    ),
    code: ({ className, children, ...props }) => {
      const match = /language-(\w+)/.exec(className || "");
      const isInline = !match && !JSON.stringify(props).includes("node");
      
      if (isInline) {
         return (
          <code
            className={cn(
              "bg-neutral-100 dark:bg-neutral-800 px-1.5 py-0.5 rounded text-sm font-mono text-neutral-800 dark:text-neutral-200",
              className
            )}
            {...props}
          >
            {children}
          </code>
        );
      }

      return (
        <pre className="bg-neutral-100 dark:bg-neutral-800 rounded-lg p-4 overflow-x-auto my-3">
          <code
            className={cn(
              "text-sm font-mono text-neutral-800 dark:text-neutral-200",
              className
            )}
            {...props}
          >
            {children}
          </code>
        </pre>
      );
    },
  };

  return (
    <div className={cn("prose prose-sm dark:prose-invert max-w-none", className)}>
      <ReactMarkdown
        remarkPlugins={[remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={components}
      >
        {cleanContent}
      </ReactMarkdown>

      {/* Image Gallery */}
      {allImages.length > 0 && (
        <div className="mt-4 not-prose">
          {allImages.length === 1 ? (
            // Single image — natural aspect, capped height
            <figure className="my-0">
              <button
                type="button"
                onClick={() => openLightbox(0)}
                className="group relative block w-full overflow-hidden rounded-xl border border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-900/50 focus:outline-none focus:ring-2 focus:ring-blue-500/40"
                aria-label={allImages[0].alt || "Open image"}
              >
                <img
                  src={allImages[0].image}
                  alt={allImages[0].alt}
                  loading="lazy"
                  className="block w-full h-auto max-h-[420px] object-cover transition-transform duration-500 group-hover:scale-[1.02]"
                />
                <span className="pointer-events-none absolute top-2 right-2 inline-flex items-center justify-center w-8 h-8 rounded-full bg-black/50 backdrop-blur-sm text-white opacity-0 group-hover:opacity-100 transition-opacity">
                  <ZoomIn className="w-4 h-4" />
                </span>
              </button>
              {allImages[0].alt && (
                <figcaption className="mt-2 text-xs text-neutral-500 dark:text-neutral-500 leading-snug">
                  {allImages[0].alt}
                </figcaption>
              )}
            </figure>
          ) : allImages.length === 2 ? (
            // Two images — side-by-side
            <div className="grid grid-cols-2 gap-2">
              {allImages.map((item, i) => (
                <GalleryThumb
                  key={i}
                  item={item}
                  onClick={() => openLightbox(i)}
                  aspect="aspect-[4/5]"
                />
              ))}
            </div>
          ) : (
            // 3+ images — tight grid
            <div className="grid grid-cols-3 gap-2">
              {allImages.map((item, i) => (
                <GalleryThumb
                  key={i}
                  item={item}
                  onClick={() => openLightbox(i)}
                  aspect="aspect-square"
                />
              ))}
            </div>
          )}
        </div>
      )}

      {/* Lightbox */}
      <Lightbox
        images={allImages}
        initialIndex={lightboxIndex}
        isOpen={lightboxOpen}
        onClose={() => setLightboxOpen(false)}
      />
    </div>
  );
}

function GalleryThumb({
  item,
  onClick,
  aspect,
}: {
  item: { image: string; alt: string };
  onClick: () => void;
  aspect: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "group relative block w-full overflow-hidden rounded-lg border border-neutral-200 dark:border-neutral-800 bg-neutral-100 dark:bg-neutral-900 focus:outline-none focus:ring-2 focus:ring-blue-500/40",
        aspect
      )}
      aria-label={item.alt || "Open image"}
    >
      <img
        src={item.image}
        alt={item.alt}
        loading="lazy"
        className="absolute inset-0 w-full h-full object-cover transition-transform duration-500 group-hover:scale-[1.04]"
      />
      <span className="pointer-events-none absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors" />
      <span className="pointer-events-none absolute top-1.5 right-1.5 inline-flex items-center justify-center w-7 h-7 rounded-full bg-black/50 backdrop-blur-sm text-white opacity-0 group-hover:opacity-100 transition-opacity">
        <ZoomIn className="w-3.5 h-3.5" />
      </span>
      {item.alt && (
        <span className="pointer-events-none absolute inset-x-0 bottom-0 px-2 py-1.5 bg-gradient-to-t from-black/70 to-transparent text-[10px] text-white/90 leading-tight truncate opacity-0 group-hover:opacity-100 transition-opacity">
          {item.alt}
        </span>
      )}
    </button>
  );
}
