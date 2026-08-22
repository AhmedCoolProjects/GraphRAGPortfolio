'use client'
import { motion } from "motion/react";
import { cn } from "@/lib/utils";
import { useRouter } from "next/navigation";
import { useStore } from "@/store/useStore";
import { NavItem } from "@/constants/navigation";

interface AnimatedNavButtonProps {
    item: NavItem;
    isActive?: boolean;
    index: number;
    onClick?: () => void;
}

function AnimatedNavButton({ item, isActive, index, onClick }: AnimatedNavButtonProps) {
    const router = useRouter();
    const { setInitialMessage } = useStore();
    const Icon = item.icon;

    const handleClick = (e: React.MouseEvent<HTMLButtonElement>) => {
        e.preventDefault();
        if (onClick) onClick();
        setInitialMessage(item.prompt);
        router.push("/chat");
    };

    return (
        <motion.li
            initial={{ opacity: 0, y: 30, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{
                duration: 0.5,
                delay: 0.8 + index * 0.1,
                ease: [0.16, 1, 0.3, 1],
            }}
            className="z-50"
        >
            <motion.button
                onClick={handleClick}
                whileHover={{ scale: 1.05, y: -4 }}
                whileTap={{ scale: 0.98 }}
                transition={{ type: "spring", stiffness: 400, damping: 17 }}
                className={cn(
                    "group relative flex h-16 w-24 sm:h-20 sm:w-28 flex-col items-center justify-center gap-1.5 rounded-xl overflow-hidden cursor-pointer",
                    "bg-white/70 dark:bg-slate-800/50",
                    "backdrop-blur-xl",
                    "border border-slate-200/50 dark:border-slate-700/50",
                    "shadow-lg shadow-slate-200/50 dark:shadow-slate-900/50",
                    "hover:shadow-xl hover:shadow-cyan-200/30 dark:hover:shadow-cyan-900/20",
                    "transition-shadow duration-300",
                    isActive && "ring-2 ring-cyan-500 dark:ring-cyan-400"
                )}
            >
                {/* Gradient Background on Hover */}
                <motion.div
                    initial={{ opacity: 0 }}
                    whileHover={{ opacity: 1 }}
                    className="absolute inset-0 bg-gradient-to-br from-cyan-50/80 via-transparent to-purple-50/80 dark:from-cyan-900/20 dark:to-purple-900/20"
                />

                {/* Shine Effect */}
                <motion.div
                    initial={{ x: '-100%', opacity: 0 }}
                    whileHover={{ x: '200%', opacity: 0.5 }}
                    transition={{ duration: 0.6, ease: "easeInOut" }}
                    className="absolute inset-0 bg-gradient-to-r from-transparent via-white/40 to-transparent skew-x-12"
                />

                {/* Icon with Animation */}
                <motion.div
                    whileHover={{ rotate: [0, -10, 10, 0] }}
                    transition={{ duration: 0.5 }}
                    className="relative z-10"
                >
                    <Icon className="h-5 w-5 sm:h-6 sm:w-6 text-cyan-600 dark:text-cyan-400 group-hover:text-cyan-700 dark:group-hover:text-cyan-300 transition-colors" />
                </motion.div>

                {/* Label */}
                <span className="relative z-10 text-xs sm:text-sm font-medium text-slate-700 dark:text-slate-300 group-hover:text-slate-900 dark:group-hover:text-white transition-colors">
                    {item.label}
                </span>

                {/* Bottom Accent Line */}
                <motion.div
                    initial={{ scaleX: 0 }}
                    whileHover={{ scaleX: 1 }}
                    transition={{ duration: 0.3 }}
                    className="absolute bottom-0 left-1/4 right-1/4 h-0.5 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-full"
                />
            </motion.button>
        </motion.li>
    );
}

interface AnimatedNavGridProps {
    items: NavItem[];
    activeSection?: string;
    onSectionChange?: (section: string) => void;
}

export function AnimatedNavGrid({ items, activeSection, onSectionChange }: AnimatedNavGridProps) {
    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.6 }}
            className="w-full flex justify-center px-4"
        >
            <ul className="flex flex-row flex-wrap items-center justify-center gap-3 sm:gap-4">
                {items.map((item, index) => (
                    <AnimatedNavButton
                        key={item.id}
                        item={item}
                        index={index}
                        isActive={activeSection === item.id}
                        onClick={() => onSectionChange?.(item.id)}
                    />
                ))}
            </ul>
        </motion.div>
    );
}
