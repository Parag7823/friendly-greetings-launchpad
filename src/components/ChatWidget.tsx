import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell } from 'recharts';
import { Button } from './ui/button';
import { Card } from './ui/card';
import { Download, ExternalLink, BarChart2, PieChart as PieChartIcon, Activity } from 'lucide-react';
import { motion } from 'framer-motion';

interface ChatWidgetProps {
    data?: any;
    visualizations?: any[];
    actions?: any[];
    onAction?: (action: any) => void;
}

const COLORS = ['#6366f1', '#8b5cf6', '#ec4899', '#10b981', '#f59e0b'];

export const ChatWidget = ({ data, visualizations, actions, onAction }: ChatWidgetProps) => {
    if (!visualizations && !actions && !data) return null;

    const renderChart = (viz: any) => {
        const ChartComponent = viz.type === 'line' ? LineChart : viz.type === 'pie' ? PieChart : BarChart;

        return (
            <div className="h-[200px] w-full mt-4 mb-2">
                <ResponsiveContainer width="100%" height="100%">
                    {viz.type === 'pie' ? (
                        <PieChart>
                            <Pie
                                data={viz.data}
                                cx="50%"
                                cy="50%"
                                innerRadius={60}
                                outerRadius={80}
                                paddingAngle={5}
                                dataKey="value"
                            >
                                {viz.data.map((_: any, index: number) => (
                                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                ))}
                            </Pie>
                            <Tooltip
                                contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #333', borderRadius: '8px' }}
                                itemStyle={{ color: '#fff' }}
                            />
                        </PieChart>
                    ) : (
                        <ChartComponent data={viz.data}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
                            <XAxis
                                dataKey="name"
                                stroke="#666"
                                fontSize={10}
                                tickLine={false}
                                axisLine={false}
                            />
                            <YAxis
                                stroke="#666"
                                fontSize={10}
                                tickLine={false}
                                axisLine={false}
                                tickFormatter={(value) => `$${value}`}
                            />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #333', borderRadius: '8px' }}
                                itemStyle={{ color: '#fff' }}
                                cursor={{ fill: '#ffffff10' }}
                            />
                            {viz.type === 'line' ? (
                                <Line
                                    type="monotone"
                                    dataKey="value"
                                    stroke="#6366f1"
                                    strokeWidth={2}
                                    dot={{ fill: '#6366f1', r: 4 }}
                                    activeDot={{ r: 6 }}
                                />
                            ) : (
                                <Bar
                                    dataKey="value"
                                    fill="#6366f1"
                                    radius={[4, 4, 0, 0]}
                                    maxBarSize={50}
                                />
                            )}
                        </ChartComponent>
                    )}
                </ResponsiveContainer>
                <p className="text-center text-xs text-muted-foreground mt-2">{viz.title}</p>
            </div>
        );
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-4 mt-3 w-full max-w-md"
        >
            {/* Visualizations */}
            {visualizations?.map((viz, idx) => (
                <Card key={idx} className="p-4 bg-black/20 border-white/10">
                    <div className="flex items-center gap-2 mb-2">
                        {viz.type === 'pie' ? <PieChartIcon className="w-4 h-4 text-primary" /> :
                            viz.type === 'line' ? <Activity className="w-4 h-4 text-primary" /> :
                                <BarChart2 className="w-4 h-4 text-primary" />}
                        <h4 className="text-xs font-semibold text-foreground">{viz.title}</h4>
                    </div>
                    {renderChart(viz)}
                </Card>
            ))}

            {/* Actions */}
            {actions && actions.length > 0 && (
                <div className="flex flex-wrap gap-2">
                    {actions.map((action, idx) => (
                        <Button
                            key={idx}
                            variant="outline"
                            size="sm"
                            onClick={() => onAction?.(action)}
                            className="text-xs h-8 bg-white/5 hover:bg-white/10 border-white/10 text-primary hover:text-primary/90"
                        >
                            {action.type === 'export' ? <Download className="w-3 h-3 mr-2" /> :
                                action.type === 'view_graph' ? <ExternalLink className="w-3 h-3 mr-2" /> : null}
                            {action.label}
                        </Button>
                    ))}
                </div>
            )}
        </motion.div>
    );
};
