"use client";

import { useEffect, useState } from "react";
import { Github, Star } from "lucide-react";

interface GithubStarButtonProps {
  repoUrl?: string;
  repoName?: string;
  className?: string;
}

export function GithubStarButton({
  repoUrl = "https://github.com/AhmedCoolProjects/GraphRAGPortfolio",
  repoName = "AhmedCoolProjects/GraphRAGPortfolio",
  className = "",
}: GithubStarButtonProps) {
  const [stars, setStars] = useState<number | null>(null);

  useEffect(() => {
    async function fetchStars() {
      try {
        const res = await fetch(`https://api.github.com/repos/${repoName}`);
        if (res.ok) {
          const data = await res.json();
          setStars(data.stargazers_count);
        }
      } catch {
        // Fallback gracefully if API rate limited or offline
      }
    }
    fetchStars();
  }, [repoName]);

  return (
    <a
      href={repoUrl}
      target="_blank"
      rel="noopener noreferrer"
      aria-label="View source code on GitHub"
      className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full 
        text-xs font-medium border border-zinc-200 dark:border-zinc-800 
        bg-zinc-50 dark:bg-zinc-900/80 hover:bg-zinc-100 dark:hover:bg-zinc-800 
        text-zinc-700 dark:text-zinc-300 hover:text-zinc-900 dark:hover:text-white 
        transition-all duration-200 shadow-sm ${className}`}
    >
      <Github size={14} className="text-zinc-700 dark:text-zinc-300" />
      <span className="hidden sm:inline font-semibold">Code</span>
      <span className="flex items-center gap-1 text-[11px] font-medium text-amber-500 dark:text-amber-400 bg-amber-50 dark:bg-amber-950/40 px-1.5 py-0.5 rounded-full border border-amber-200 dark:border-amber-800/50">
        <Star size={11} className="fill-amber-400 text-amber-400" />
        {stars !== null ? stars : "Star"}
      </span>
    </a>
  );
}
