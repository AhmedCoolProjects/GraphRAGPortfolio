'use client'
import { motion, Variants } from "motion/react";
import { PointerHighlight } from "@/components/ui/pointer-highlight";

export function AnimatedHeroIntro() {
    const containerVariants: Variants = {
        hidden: { opacity: 0 },
        visible: {
            opacity: 1,
            transition: {
                staggerChildren: 0.1,
                delayChildren: 0.2,
            },
        },
    };

    const itemVariants: Variants = {
        hidden: { opacity: 0, y: 20 },
        visible: {
            opacity: 1,
            y: 0,
            transition: {
                duration: 0.6,
                ease: "easeOut",
            },
        },
    };

    return (
        <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            className="text-center lg:text-left"
        >
            {/* Greeting Line */}
            <motion.div 
                variants={itemVariants}
                className="mb-2"
            >
                <motion.span 
                    className="inline-block text-2xl sm:text-3xl lg:text-4xl font-light text-slate-600 dark:text-slate-400"
                    animate={{ opacity: [0.7, 1, 0.7] }}
                    transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
                >
                    Hey, I&apos;m
                </motion.span>
            </motion.div>

            {/* Name with Gradient */}
            <motion.div variants={itemVariants} className="mb-4">
                <motion.h1 
                    className="text-4xl sm:text-5xl lg:text-6xl font-bold bg-gradient-to-r from-cyan-600 via-blue-600 to-purple-600 dark:from-cyan-400 dark:via-blue-400 dark:to-purple-400 bg-clip-text text-transparent"
                    whileHover={{ scale: 1.02 }}
                    transition={{ type: "spring", stiffness: 400, damping: 10 }}
                >
                    Ahmed Bargady
                </motion.h1>
            </motion.div>

            {/* Tagline with Animated Highlights */}
            <motion.div 
                variants={itemVariants}
                className="text-lg sm:text-xl lg:text-2xl text-slate-700 dark:text-slate-300 flex flex-wrap justify-center lg:justify-start items-center gap-1"
            >
                <span>I do</span>
                <PointerHighlight
                    rectangleClassName="bg-cyan-100/80 dark:bg-cyan-900/50 border-cyan-300 dark:border-cyan-700"
                    pointerClassName="text-cyan-500 h-3 w-3"
                    containerClassName="inline-block mx-1"
                >
                    <span className="relative z-10 font-semibold text-cyan-700 dark:text-cyan-300 px-1">
                        AI Research
                    </span>
                </PointerHighlight>
                <span>for</span>
                <PointerHighlight
                    rectangleClassName="bg-rose-100/80 dark:bg-rose-900/50 border-rose-300 dark:border-rose-700"
                    pointerClassName="text-rose-500 h-3 w-3"
                    containerClassName="inline-block mx-1"
                >
                    <span className="relative z-10 font-semibold text-rose-700 dark:text-rose-300 px-1">
                        Cyber Security
                    </span>
                </PointerHighlight>
            </motion.div>

            {/* Subtitle */}
            <motion.p 
                variants={itemVariants}
                className="mt-4 text-sm sm:text-base text-slate-500 dark:text-slate-500 max-w-md mx-auto lg:mx-0"
            >
                PhD Student at UM6P • Research Engineer • AI Enthusiast
            </motion.p>

            {/* Decorative Line */}
            <motion.div 
                variants={itemVariants}
                className="mt-6 flex justify-center lg:justify-start"
            >
                <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: 80 }}
                    transition={{ duration: 1, delay: 1, ease: "easeOut" }}
                    className="h-0.5 bg-gradient-to-r from-transparent via-cyan-500/50 to-transparent"
                />
            </motion.div>
        </motion.div>
    );
}
