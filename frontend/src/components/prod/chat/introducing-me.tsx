"use client";

import Image from "next/image";
import { KNOWLEDGE_BASE } from "@/constants/knowledge-base";

export function IntroducingMe() {
  const kb = KNOWLEDGE_BASE;

  return (
    <div className="w-full max-w-4xl mx-auto">
      {/* Profile Card */}
      <div className="bg-white dark:bg-neutral-900 rounded-3xl shadow-lg border border-black/10 dark:border-white/10 p-6 sm:p-8">
        <div className="flex flex-col md:flex-row gap-6 sm:gap-8">
          {/* Left: Profile Picture */}
          <div className="flex-shrink-0">
            <div className="relative w-48 h-48 sm:w-64 sm:h-64 mx-auto md:mx-0">
              <Image
                src="/me.png"
                alt={kb.personal.name}
                fill
                className="object-cover rounded-2xl dark:invert"
                priority
              />
            </div>
          </div>

          {/* Right: Profile Info */}
          <div className="flex-1 space-y-4">
            {/* Name & Basic Info */}
            <div>
              <h1 className="text-3xl sm:text-4xl font-bold text-neutral-900 dark:text-white">
                {kb.personal.name}
              </h1>
              <div className="flex flex-wrap gap-3 mt-2 text-sm text-neutral-600 dark:text-neutral-400">
                {kb.personal.age && <span>{kb.personal.age} years old</span>}
                <span>•</span>
                <span>{kb.personal.location}</span>
              </div>
            </div>

            {/* Bio */}
            <div className="space-y-2">
              <p className="text-base text-neutral-700 dark:text-neutral-300 leading-relaxed">
                {kb.personal.tagline}
              </p>
              <p className="text-sm text-neutral-600 dark:text-neutral-400 leading-relaxed">
                {kb.personal.bio}
              </p>
            </div>

            {/* Tags */}
            <div className="flex flex-wrap gap-2">
              {kb.personal.tags.map((tag, index) => (
                <span
                  key={index}
                  className="px-4 py-1.5 bg-blue-500 text-white text-sm font-medium rounded-full hover:bg-blue-600 transition-colors"
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* Expanded Bio Section */}
        <div className="mt-8 pt-6 border-t border-black/10 dark:border-white/10">
          <p className="text-sm sm:text-base text-neutral-700 dark:text-neutral-300 leading-relaxed">
            I&apos;m {kb.personal.name}, {kb.personal.title} at{" "}
            {kb.education[0]?.institution}. My journey in technology has been
            driven by passion for {kb.interests.slice(0, 3).join(", ")}, and
            creating innovative solutions. I focus on{" "}
            {kb.skills[0]?.items.slice(0, 3).join(", ")}, and I&apos;m always
            exploring new frontiers in technology.
          </p>

          {/* Call to Action */}
          <div className="mt-6">
            <p className="text-sm font-medium text-neutral-900 dark:text-white mb-3">
              What aspect of my journey interests you the most?
            </p>
            <div className="text-xs text-neutral-500 dark:text-neutral-400">
              Ask me about my projects, skills, education, or anything else!
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
