'use client'
import { useEffect, useState } from "react";
import { motion } from "motion/react";

const StreamColumn = ({ delay, x, duration, color }: { delay: number, x: string, duration: number, color: string }) => {
    return (
        <motion.div
            initial={{ y: -100, opacity: 0 }}
            animate={{
                y: ['0%', '100%'],
                opacity: [0, 1, 0]
            }}
            transition={{
                duration: duration,
                repeat: Infinity,
                delay: delay,
                ease: "linear"
            }}
            className={`absolute top-0 w-[1px] h-[30vh] bg-gradient-to-b from-transparent ${color} to-transparent`}
            style={{ left: x }}
        />
    );
};

export function CyberStreamBackground() {
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        setMounted(true);
    }, []);

    if (!mounted) return null;

    return (
        <div className="absolute inset-0 z-0 pointer-events-none overflow-hidden bg-slate-950">

            {/* Base Grid - Subtle */}
            <div
                className="absolute inset-0 opacity-[0.05]"
                style={{
                    backgroundImage: `linear-gradient(#06b6d4 1px, transparent 1px), linear-gradient(to right, #06b6d4 1px, transparent 1px)`,
                    backgroundSize: '40px 40px'
                }}
            />

            {/* Cyber Streams (Rain Effect) */}
            <div className="absolute inset-0 opacity-20 dark:opacity-30">
                {/* Generate random streams - Reduced count for better performance on mobile */}
                {Array.from({ length: 12 }).map((_, i) => (
                    <StreamColumn
                        key={`stream-${i}`}
                        delay={i * 0.8}
                        x={`${(i + 1) * 8}%`}
                        duration={5 + (i % 3)}
                        color={i % 2 === 0 ? "via-cyan-500" : "via-indigo-500"}
                    />
                ))}
            </div>

            {/* Animated Glowing Orbs (Focus Points) */}
            <motion.div
                animate={{
                    scale: [1, 1.2, 1],
                    opacity: [0.3, 0.5, 0.3],
                }}
                transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
                className="absolute top-1/4 left-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-[100px]"
            />

            <motion.div
                animate={{
                    scale: [1.2, 1, 1.2],
                    opacity: [0.2, 0.4, 0.2],
                }}
                transition={{ duration: 10, repeat: Infinity, ease: "easeInOut", delay: 2 }}
                className="absolute bottom-1/4 right-1/4 w-[500px] h-[500px] bg-indigo-500/10 rounded-full blur-[120px]"
            />

            {/* Horizontal Scanning Line */}
            <motion.div
                animate={{ top: ['0%', '100%'] }}
                transition={{ duration: 15, repeat: Infinity, ease: "linear" }}
                className="absolute left-0 w-full h-[1px] bg-gradient-to-r from-transparent via-cyan-400/30 to-transparent blur-[1px]"
            />
        </div>
    );
}
