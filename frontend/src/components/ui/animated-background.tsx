'use client'
import { motion } from "motion/react";
import { useEffect, useState } from "react";

// Configuration constants for animations
const PARTICLE_COUNT = 20;
const SHAPE_COUNT = 8;

interface Particle {
    id: number;
    x: number;
    y: number;
    size: number;
    duration: number;
    delay: number;
}

interface FloatingShape {
    id: number;
    x: number;
    y: number;
    size: number;
    rotation: number;
    duration: number;
    delay: number;
    type: 'circle' | 'square' | 'triangle' | 'hexagon';
    opacity: number;
}

const FloatingShapeComponent = ({ shape }: { shape: FloatingShape }) => {
    const getShapeClasses = () => {
        const baseClasses = "absolute border border-cyan-500/20 dark:border-cyan-400/15";
        switch (shape.type) {
            case 'circle':
                return `${baseClasses} rounded-full`;
            case 'square':
                return `${baseClasses} rounded-lg`;
            case 'triangle':
                return baseClasses;
            case 'hexagon':
                return `${baseClasses} rounded-xl`;
            default:
                return baseClasses;
        }
    };

    return (
        <motion.div
            className={getShapeClasses()}
            style={{
                width: shape.size,
                height: shape.size,
                left: `${shape.x}%`,
                top: `${shape.y}%`,
            }}
            initial={{ 
                opacity: 0, 
                scale: 0,
                rotate: 0,
            }}
            animate={{ 
                opacity: [0, shape.opacity, shape.opacity, 0],
                scale: [0.5, 1, 1, 0.5],
                rotate: [0, shape.rotation, shape.rotation * 2, shape.rotation * 3],
                y: [0, -30, -60, -90],
            }}
            transition={{
                duration: shape.duration,
                delay: shape.delay,
                repeat: Infinity,
                ease: "easeInOut",
            }}
        />
    );
};

const ParticleComponent = ({ particle }: { particle: Particle }) => {
    return (
        <motion.div
            className="absolute rounded-full bg-gradient-to-br from-cyan-400/40 to-blue-500/40 dark:from-cyan-400/30 dark:to-blue-500/30"
            style={{
                width: particle.size,
                height: particle.size,
                left: `${particle.x}%`,
                top: `${particle.y}%`,
            }}
            initial={{ opacity: 0, scale: 0 }}
            animate={{
                opacity: [0, 0.8, 0.8, 0],
                scale: [0, 1, 1, 0],
                y: [0, -100, -200, -300],
            }}
            transition={{
                duration: particle.duration,
                delay: particle.delay,
                repeat: Infinity,
                ease: "easeOut",
            }}
        />
    );
};

export function AnimatedBackground() {
    const [mounted, setMounted] = useState(false);
    const [particles, setParticles] = useState<Particle[]>([]);
    const [shapes, setShapes] = useState<FloatingShape[]>([]);

    useEffect(() => {
        setMounted(true);
        
        // Generate particles
        const generatedParticles = Array.from({ length: PARTICLE_COUNT }).map((_, i) => ({
            id: i,
            x: Math.random() * 100,
            y: 50 + Math.random() * 50,
            size: 2 + Math.random() * 4,
            duration: 8 + Math.random() * 8,
            delay: Math.random() * 10,
        }));
        setParticles(generatedParticles);

        // Generate floating shapes
        const shapeTypes: FloatingShape['type'][] = ['circle', 'square', 'triangle', 'hexagon'];
        const generatedShapes = Array.from({ length: SHAPE_COUNT }).map((_, i) => ({
            id: i,
            x: 10 + Math.random() * 80,
            y: 20 + Math.random() * 60,
            size: 40 + Math.random() * 80,
            rotation: 90 + Math.random() * 180,
            duration: 15 + Math.random() * 10,
            delay: Math.random() * 5,
            type: shapeTypes[Math.floor(Math.random() * shapeTypes.length)],
            opacity: 0.1 + Math.random() * 0.15,
        }));
        setShapes(generatedShapes);
    }, []);

    if (!mounted) return null;

    return (
        <div className="fixed inset-0 z-0 pointer-events-none overflow-hidden">
            {/* Gradient Background */}
            <div className="absolute inset-0 bg-gradient-to-br from-slate-50 via-white to-slate-100 dark:from-slate-950 dark:via-slate-900 dark:to-slate-950" />
            
            {/* Mesh Gradient Blobs */}
            <motion.div
                animate={{
                    scale: [1, 1.2, 1],
                    opacity: [0.3, 0.5, 0.3],
                    x: [0, 50, 0],
                }}
                transition={{ duration: 20, repeat: Infinity, ease: "easeInOut" }}
                className="absolute -top-40 -left-40 w-[600px] h-[600px] bg-gradient-to-br from-cyan-300/30 to-blue-400/20 dark:from-cyan-500/20 dark:to-blue-600/10 rounded-full blur-[100px]"
            />
            
            <motion.div
                animate={{
                    scale: [1.2, 1, 1.2],
                    opacity: [0.2, 0.4, 0.2],
                    x: [0, -50, 0],
                }}
                transition={{ duration: 25, repeat: Infinity, ease: "easeInOut", delay: 5 }}
                className="absolute -bottom-40 -right-40 w-[700px] h-[700px] bg-gradient-to-br from-purple-300/25 to-pink-400/15 dark:from-purple-500/15 dark:to-pink-600/10 rounded-full blur-[120px]"
            />
            
            <motion.div
                animate={{
                    scale: [1, 1.3, 1],
                    opacity: [0.15, 0.3, 0.15],
                }}
                transition={{ duration: 18, repeat: Infinity, ease: "easeInOut", delay: 3 }}
                className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-gradient-to-br from-indigo-300/20 to-cyan-400/15 dark:from-indigo-500/10 dark:to-cyan-600/5 rounded-full blur-[100px]"
            />

            {/* Subtle Grid Pattern */}
            <div
                className="absolute inset-0 opacity-[0.015] dark:opacity-[0.03]"
                style={{
                    backgroundImage: `
                        linear-gradient(rgba(6, 182, 212, 0.5) 1px, transparent 1px),
                        linear-gradient(90deg, rgba(6, 182, 212, 0.5) 1px, transparent 1px)
                    `,
                    backgroundSize: '60px 60px',
                }}
            />

            {/* Floating Geometric Shapes */}
            {shapes.map((shape) => (
                <FloatingShapeComponent key={`shape-${shape.id}`} shape={shape} />
            ))}

            {/* Particles */}
            {particles.map((particle) => (
                <ParticleComponent key={`particle-${particle.id}`} particle={particle} />
            ))}

            {/* Animated Gradient Lines */}
            <motion.div
                animate={{ 
                    x: ['-100%', '200%'],
                }}
                transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
                className="absolute top-1/4 left-0 w-full h-[1px] bg-gradient-to-r from-transparent via-cyan-400/30 to-transparent"
            />
            
            <motion.div
                animate={{ 
                    x: ['200%', '-100%'],
                }}
                transition={{ duration: 10, repeat: Infinity, ease: "linear", delay: 2 }}
                className="absolute top-3/4 left-0 w-full h-[1px] bg-gradient-to-r from-transparent via-purple-400/20 to-transparent"
            />

            {/* Corner Decorations */}
            <div className="absolute top-0 left-0 w-32 h-32 border-l-2 border-t-2 border-cyan-500/10 dark:border-cyan-400/5 rounded-tl-3xl" />
            <div className="absolute top-0 right-0 w-32 h-32 border-r-2 border-t-2 border-cyan-500/10 dark:border-cyan-400/5 rounded-tr-3xl" />
            <div className="absolute bottom-0 left-0 w-32 h-32 border-l-2 border-b-2 border-purple-500/10 dark:border-purple-400/5 rounded-bl-3xl" />
            <div className="absolute bottom-0 right-0 w-32 h-32 border-r-2 border-b-2 border-purple-500/10 dark:border-purple-400/5 rounded-br-3xl" />
        </div>
    );
}
