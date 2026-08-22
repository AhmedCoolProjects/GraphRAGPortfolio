"use client";

import { cn } from "@/lib/utils";
import { useRouter } from "next/navigation";
import { useStore } from "@/store/useStore";

interface NavButtonProps {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  prompt: string;
  isActive?: boolean;
  onClick?: () => void;
}

export function NavButton({ icon: Icon, label, prompt, isActive, onClick }: NavButtonProps) {
  const router = useRouter();
  const { setInitialMessage } = useStore();

  const handleClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    e.preventDefault();

    // Call the original onClick if provided
    if (onClick) {
      onClick();
    }

    // Set the prompt as initial message
    setInitialMessage(prompt);

    // Redirect to chat page
    router.push("/chat");
  };

  return (
    <li className="z-50">
      <button
        onClick={handleClick}
        className={cn(
          "group flex h-16 w-24 sm:h-20 sm:w-32 flex-col items-center justify-center gap-1 sm:gap-2 rounded-2xl border-0.5 border-black/10 dark:border-white/10 bg-white/60 dark:bg-white/5 backdrop-blur-md shadow-sm hover:shadow-md ring-1 ring-black/5 dark:ring-white/10 transition-transform duration-200 hover:scale-[1.02] cursor-pointer",
          isActive && "ring-2 ring-blue-500 dark:ring-blue-400"
        )}
      >
        <Icon className="h-4 w-4 sm:h-5 sm:w-5 text-blue-600 dark:text-blue-400" />
        <span className="text-xs sm:text-sm text-neutral-800 dark:text-neutral-200 font-mono">
          {label}
        </span>
      </button>
    </li>
  );
}
