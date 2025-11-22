import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
    Bot,
    ArrowRight,
    TrendingUp,
    AlertCircle,
    CheckCircle2,
    Zap,
    X
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { useAuth } from '@/components/AuthProvider';
import { ChatWidget } from './ChatWidget';

interface AidentGreetingProps {
    hasConnections: boolean;
    onConnectClick: () => void;
}

export const AidentGreeting: React.FC<AidentGreetingProps> = ({
    hasConnections,
    onConnectClick
}) => {
    const { user } = useAuth();
    const [activeVisualization, setActiveVisualization] = useState<any>(null);
    const timeOfDay = new Date().getHours() < 12 ? 'Good morning' : new Date().getHours() < 18 ? 'Good afternoon' : 'Good evening';

    useEffect(() => {
        const handleVisualize = (event: CustomEvent) => {
            setActiveVisualization(event.detail);
        };

        window.addEventListener('visualize-data', handleVisualize as EventListener);
        return () => window.removeEventListener('visualize-data', handleVisualize as EventListener);
    }, []);

    // ANIMATIONS
    const containerVariants = {
        hidden: { opacity: 0 },
        visible: {
            opacity: 1,
            transition: {
                staggerChildren: 0.1,
                delayChildren: 0.2
            }
        }
    };

    const itemVariants = {
        hidden: { y: 20, opacity: 0 },
        visible: {
            y: 0,
            opacity: 1,
            transition: { type: "spring", stiffness: 300, damping: 24 }
        }
    };

    // --- STATE 1: NEW USER (No Connections) ---
    if (!hasConnections) {
        return (
            <div className="h-full flex flex-col items-center justify-center p-8 bg-gradient-to-b from-background to-slate-900/50">
                <motion.div
                    initial="hidden"
                    animate="visible"
                    variants={containerVariants}
                    className="max-w-2xl w-full text-center space-y-8"
                >
                    {/* Avatar / Logo */}
                    <motion.div variants={itemVariants} className="flex justify-center">
                        <div className="relative">
                            <div className="absolute inset-0 bg-primary/20 blur-3xl rounded-full" />
                            <div className="relative bg-card border border-border p-6 rounded-2xl shadow-2xl shadow-primary/10">
                                <Bot className="w-16 h-16 text-primary" />
                            </div>
                        </div>
                    </motion.div>

                    {/* Headline */}
                    <motion.div variants={itemVariants} className="space-y-4">
                        <h1 className="text-4xl md:text-5xl font-bold tracking-tight text-foreground">
                            I'm <span className="text-primary">Aident</span>.
                        </h1>
                        <p className="text-xl text-muted-foreground max-w-lg mx-auto leading-relaxed">
                            Your intelligent finance employee. Connect your accounting software, and I'll start analyzing your cash flow immediately.
                        </p>
                    </motion.div>

                    {/* Value Props */}
                    <motion.div variants={itemVariants} className="grid grid-cols-1 md:grid-cols-3 gap-4 text-left">
                        {[
                            { icon: TrendingUp, title: "Cash Flow", desc: "Real-time forecasting" },
                            { icon: AlertCircle, title: "Audit", desc: "Detect anomalies instantly" },
                            { icon: Zap, title: "Insights", desc: "Proactive financial advice" }
                        ].map((item, i) => (
                            <Card key={i} className="p-4 bg-card/50 border-border/50 hover:border-primary/50 transition-colors">
                                <div className="flex items-center gap-3 mb-2">
                                    <div className="p-2 bg-primary/10 rounded-lg">
                                        <item.icon className="w-4 h-4 text-primary" />
                                    </div>
                                    <span className="font-semibold text-foreground">{item.title}</span>
                                </div>
                                <p className="text-sm text-muted-foreground">{item.desc}</p>
                            </Card>
                        ))}
                    </motion.div>

                    {/* CTA */}
                    <motion.div variants={itemVariants} className="pt-8">
                        <Button
                            size="lg"
                            onClick={onConnectClick}
                            className="bg-primary hover:bg-primary/90 text-white px-8 py-6 text-lg rounded-xl shadow-lg shadow-primary/20 hover:shadow-primary/40 transition-all hover:scale-105 group"
                        >
                            Connect Data Sources
                            <ArrowRight className="ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform" />
                        </Button>
                        <p className="mt-4 text-sm text-muted-foreground">
                            Securely connects with QuickBooks, Xero, Stripe & more.
                        </p>
                    </motion.div>
                </motion.div>
            </div>
        );
    }

    // --- STATE 2: RETURNING USER (Has Connections) ---
    return (
        <div className="h-full flex flex-col p-8 overflow-y-auto bg-gradient-to-b from-background to-slate-900/20">
            <motion.div
                initial="hidden"
                animate="visible"
                variants={containerVariants}
                className="max-w-5xl w-full mx-auto space-y-8"
            >
                {/* Header */}
                <motion.div variants={itemVariants} className="flex items-center justify-between">
                    <div>
                        <h1 className="text-3xl font-bold text-foreground">
                            {timeOfDay}, {user?.email?.split('@')[0] || 'Chief'}.
                        </h1>
                        <p className="text-muted-foreground mt-1">
                            Here is your financial briefing for today.
                        </p>
                    </div>
                    <div className="flex items-center gap-2 px-3 py-1 bg-green-500/10 border border-green-500/20 rounded-full">
                        <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                        <span className="text-xs font-medium text-green-500">Systems Online</span>
                    </div>
                </motion.div>

                {/* Daily Briefing Cards */}
                <motion.div variants={itemVariants} className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    {/* Card 1: Cash Flow */}
                    <Card className="p-6 bg-card border-border hover:border-primary/30 transition-all cursor-pointer group">
                        <div className="flex justify-between items-start mb-4">
                            <div className="p-2 bg-blue-500/10 rounded-lg">
                                <TrendingUp className="w-5 h-5 text-blue-500" />
                            </div>
                            <span className="text-xs font-medium text-green-500 flex items-center">
                                +12% <ArrowRight className="w-3 h-3 ml-1 rotate-[-45deg]" />
                            </span>
                        </div>
                        <h3 className="text-sm font-medium text-muted-foreground">Cash Flow Forecast</h3>
                        <p className="text-2xl font-bold text-foreground mt-1">$142,300</p>
                        <p className="text-xs text-muted-foreground mt-2 group-hover:text-primary transition-colors">
                            View detailed forecast &rarr;
                        </p>
                    </Card>

                    {/* Card 2: Anomalies */}
                    <Card className="p-6 bg-card border-border hover:border-primary/30 transition-all cursor-pointer group">
                        <div className="flex justify-between items-start mb-4">
                            <div className="p-2 bg-amber-500/10 rounded-lg">
                                <AlertCircle className="w-5 h-5 text-amber-500" />
                            </div>
                            <span className="text-xs font-medium text-amber-500">Action Required</span>
                        </div>
                        <h3 className="text-sm font-medium text-muted-foreground">Anomalies Detected</h3>
                        <p className="text-2xl font-bold text-foreground mt-1">3 Items</p>
                        <p className="text-xs text-muted-foreground mt-2 group-hover:text-primary transition-colors">
                            Review flagged transactions &rarr;
                        </p>
                    </Card>

                    {/* Card 3: Sync Status */}
                    <Card className="p-6 bg-card border-border hover:border-primary/30 transition-all cursor-pointer group">
                        <div className="flex justify-between items-start mb-4">
                            <div className="p-2 bg-primary/10 rounded-lg">
                                <CheckCircle2 className="w-5 h-5 text-primary" />
                            </div>
                            <span className="text-xs font-medium text-muted-foreground">Just now</span>
                        </div>
                        <h3 className="text-sm font-medium text-muted-foreground">Data Sync</h3>
                        <p className="text-2xl font-bold text-foreground mt-1">Up to Date</p>
                        <p className="text-xs text-muted-foreground mt-2 group-hover:text-primary transition-colors">
                            QuickBooks, Stripe, Xero &rarr;
                        </p>
                    </Card>
                </motion.div>

                {/* Recent Insights / "Whiteboard" Area */}
                <motion.div variants={itemVariants} className="space-y-4">
                    <div className="flex items-center justify-between">
                        <h2 className="text-lg font-semibold text-foreground">
                            {activeVisualization ? "Analysis Result" : "Recent Insights"}
                        </h2>
                        {activeVisualization && (
                            <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => setActiveVisualization(null)}
                                className="h-8 px-2 text-muted-foreground hover:text-foreground"
                            >
                                <X className="w-4 h-4 mr-1" />
                                Clear
                            </Button>
                        )}
                    </div>

                    <Card className={`p-6 bg-card/50 border-dashed border-border min-h-[300px] flex flex-col ${activeVisualization ? 'items-start' : 'items-center justify-center text-center'}`}>
                        {activeVisualization ? (
                            <div className="w-full h-full">
                                <ChatWidget
                                    visualizations={activeVisualization.visualizations || [activeVisualization]}
                                    data={activeVisualization.data}
                                    actions={activeVisualization.actions}
                                />
                            </div>
                        ) : (
                            <>
                                <div className="p-4 bg-muted rounded-full mb-4">
                                    <Bot className="w-8 h-8 text-muted-foreground" />
                                </div>
                                <h3 className="text-lg font-medium text-foreground">Ready to analyze</h3>
                                <p className="text-muted-foreground max-w-md mt-2">
                                    Ask me anything in the chat on the left, or select a card above to dive deeper.
                                </p>
                                <div className="flex gap-2 mt-6">
                                    {["Analyze Q3 Expenses", "Forecast next month", "Show burn rate"].map((query) => (
                                        <Button
                                            key={query}
                                            variant="outline"
                                            className="text-xs rounded-full hover:bg-primary/10 hover:text-primary hover:border-primary/30"
                                        >
                                            {query}
                                        </Button>
                                    ))}
                                </div>
                            </>
                        )}
                    </Card>
                </motion.div>
            </motion.div>
        </div>
    );
};
