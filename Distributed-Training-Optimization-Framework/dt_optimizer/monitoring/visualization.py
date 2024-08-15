import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Callable
from dataclasses import dataclass, field
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

@dataclass
class PlotConfig:
    title: str
    xlabel: str
    ylabel: str
    plot_type: str = "line"
    color: str = "blue"
    style: str = "-"
    marker: str = "o"
    alpha: float = 0.7
    figsize: Tuple[int, int] = (10, 6)
    grid: bool = True
    legend: bool = True
    custom_style: Optional[Dict] = None

class AdvancedVisualizer:
    def __init__(self, style: str = "default", use_plotly: bool = False):
        self.figures: Dict[str, Union[plt.Figure, go.Figure]] = {}
        self.axes: Dict[str, plt.Axes] = {}
        self.style = style
        self.use_plotly = use_plotly
        self._set_style()

    def _set_style(self):
        if not self.use_plotly:
            plt.style.use(self.style)
            sns.set_style(self.style)

    def plot_metric(self, metric_name: str, values: List[float], config: PlotConfig) -> None:
        if self.use_plotly:
            self._plot_metric_plotly(metric_name, values, config)
        else:
            self._plot_metric_matplotlib(metric_name, values, config)

    def _plot_metric_matplotlib(self, metric_name: str, values: List[float], config: PlotConfig) -> None:
        if metric_name not in self.figures:
            fig, ax = plt.subplots(figsize=config.figsize)
            self.figures[metric_name] = fig
            self.axes[metric_name] = ax
        ax = self.axes[metric_name]
        
        x = range(1, len(values) + 1)
        if config.plot_type == "line":
            ax.plot(x, values, label=metric_name, color=config.color, linestyle=config.style, marker=config.marker, alpha=config.alpha)
        elif config.plot_type == "scatter":
            ax.scatter(x, values, label=metric_name, color=config.color, marker=config.marker, alpha=config.alpha)
        elif config.plot_type == "bar":
            ax.bar(x, values, label=metric_name, color=config.color, alpha=config.alpha)
        
        ax.set_xlabel(config.xlabel)
        ax.set_ylabel(config.ylabel)
        ax.set_title(config.title)
        if config.grid:
            ax.grid(True)
        if config.legend:
            ax.legend()
        
        if config.custom_style:
            ax.set(**config.custom_style)

    def _plot_metric_plotly(self, metric_name: str, values: List[float], config: PlotConfig) -> None:
        if metric_name not in self.figures:
            self.figures[metric_name] = go.Figure()
        fig = self.figures[metric_name]
        
        x = list(range(1, len(values) + 1))
        if config.plot_type == "line":
            fig.add_trace(go.Scatter(x=x, y=values, mode='lines+markers', name=metric_name,
                                     line=dict(color=config.color, dash=config.style),
                                     marker=dict(symbol=config.marker, size=8)))
        elif config.plot_type == "scatter":
            fig.add_trace(go.Scatter(x=x, y=values, mode='markers', name=metric_name,
                                     marker=dict(color=config.color, symbol=config.marker, size=8)))
        elif config.plot_type == "bar":
            fig.add_trace(go.Bar(x=x, y=values, name=metric_name, marker_color=config.color))
        
        fig.update_layout(
            title=config.title,
            xaxis_title=config.xlabel,
            yaxis_title=config.ylabel,
            showlegend=config.legend,
            width=config.figsize[0]*100,
            height=config.figsize[1]*100
        )
        
        if config.grid:
            fig.update_xaxes(showgrid=True)
            fig.update_yaxes(showgrid=True)

    def plot_confusion_matrix(self, cm: np.ndarray, classes: List[str], normalize: bool = False, title: str = 'Confusion matrix') -> None:
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        if self.use_plotly:
            fig = go.Figure(data=go.Heatmap(z=cm, x=classes, y=classes, colorscale='Viridis'))
            fig.update_layout(title=title, xaxis_title='Predicted label', yaxis_title='True label')
            self.figures['confusion_matrix'] = fig
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='viridis', ax=ax)
            ax.set_xlabel('Predicted label')
            ax.set_ylabel('True label')
            ax.set_title(title)
            self.figures['confusion_matrix'] = fig

    def plot_distribution(self, data: List[float], title: str = 'Distribution', bins: int = 30) -> None:
        if self.use_plotly:
            fig = go.Figure(data=[go.Histogram(x=data, nbinsx=bins)])
            fig.update_layout(title=title, xaxis_title='Value', yaxis_title='Frequency')
            self.figures['distribution'] = fig
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data, bins=bins, kde=True, ax=ax)
            ax.set_title(title)
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            self.figures['distribution'] = fig

    def plot_correlation_matrix(self, data: pd.DataFrame, title: str = 'Correlation Matrix') -> None:
        corr = data.corr()
        if self.use_plotly:
            fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.index, y=corr.columns, colorscale='RdBu'))
            fig.update_layout(title=title)
            self.figures['correlation_matrix'] = fig
        else:
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            ax.set_title(title)
            self.figures['correlation_matrix'] = fig

    def plot_learning_curve(self, train_scores: List[float], val_scores: List[float], train_sizes: List[int], title: str = 'Learning Curve') -> None:
        if self.use_plotly:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=train_sizes, y=np.mean(train_scores, axis=1), mode='lines+markers', name='Training score'))
            fig.add_trace(go.Scatter(x=train_sizes, y=np.mean(val_scores, axis=1), mode='lines+markers', name='Cross-validation score'))
            fig.update_layout(title=title, xaxis_title='Training examples', yaxis_title='Score')
            self.figures['learning_curve'] = fig
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training score')
            ax.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', label='Cross-validation score')
            ax.set_title(title)
            ax.set_xlabel('Training examples')
            ax.set_ylabel('Score')
            ax.legend(loc='best')
            ax.grid(True)
            self.figures['learning_curve'] = fig

    def plot_feature_importance(self, importance: np.ndarray, features: List[str], title: str = 'Feature Importance') -> None:
        sorted_idx = importance.argsort()
        if self.use_plotly:
            fig = go.Figure(go.Bar(y=[features[i] for i in sorted_idx], x=importance[sorted_idx], orientation='h'))
            fig.update_layout(title=title, xaxis_title='Importance', yaxis_title='Features')
            self.figures['feature_importance'] = fig
        else:
            fig, ax = plt.subplots(figsize=(10, len(features) // 3))
            ax.barh([features[i] for i in sorted_idx], importance[sorted_idx])
            ax.set_title(title)
            ax.set_xlabel('Importance')
            ax.set_ylabel('Features')
            self.figures['feature_importance'] = fig

    def plot_roc_curve(self, fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, title: str = 'Receiver Operating Characteristic (ROC) Curve') -> None:
        if self.use_plotly:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.2f})'))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
            fig.update_layout(title=title, xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
            self.figures['roc_curve'] = fig
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], 'k--', label='Random')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(title)
            ax.legend(loc="lower right")
            self.figures['roc_curve'] = fig

    def plot_pca(self, data: np.ndarray, n_components: int = 2, title: str = 'PCA Visualization') -> None:
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(data)
        
        if self.use_plotly:
            fig = go.Figure(data=[go.Scatter(x=pca_result[:, 0], y=pca_result[:, 1], mode='markers')])
            fig.update_layout(title=title, xaxis_title='First Principal Component', yaxis_title='Second Principal Component')
            self.figures['pca'] = fig
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(pca_result[:, 0], pca_result[:, 1])
            ax.set_xlabel('First Principal Component')
            ax.set_ylabel('Second Principal Component')
            ax.set_title(title)
            self.figures['pca'] = fig

    def plot_tsne(self, data: np.ndarray, n_components: int = 2, perplexity: int = 30, title: str = 't-SNE Visualization') -> None:
        tsne = TSNE(n_components=n_components, perplexity=perplexity)
        tsne_result = tsne.fit_transform(data)
        
        if self.use_plotly:
            fig = go.Figure(data=[go.Scatter(x=tsne_result[:, 0], y=tsne_result[:, 1], mode='markers')])
            fig.update_layout(title=title, xaxis_title='t-SNE feature 1', yaxis_title='t-SNE feature 2')
            self.figures['tsne'] = fig
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(tsne_result[:, 0], tsne_result[:, 1])
            ax.set_xlabel('t-SNE feature 1')
            ax.set_ylabel('t-SNE feature 2')
            ax.set_title(title)
            self.figures['tsne'] = fig

    def save_plot(self, plot_name: str, filename: str) -> None:
        if plot_name in self.figures:
            if self.use_plotly:
                self.figures[plot_name].write_image(filename)
            else:
                self.figures[plot_name].savefig(filename)
            print(f"Plot saved as {filename}")
        else:
            print(f"No plot found with name: {plot_name}")

    def show_plots(self) -> None:
        if self.use_plotly:
            for fig in self.figures.values():
                fig.show()
        else:
            plt.show()

    def reset(self) -> None:
        self.figures = {}
        self.axes = {}

# Example usage
if __name__ == "__main__":
    visualizer = AdvancedVisualizer(style="darkgrid", use_plotly=True)

    # Example data
    epochs = range(1, 11)
    train_loss = [0.8, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.33, 0.3]
    val_loss = [0.9, 0.8, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.43, 0.4]

    # Plot training and validation loss
    config = PlotConfig(title="Training and Validation Loss", xlabel="Epoch", ylabel="Loss", plot_type="line")
    visualizer.plot_metric("train_loss", train_loss, config)
    config.color = "red"
    visualizer.plot_metric("val_loss", val_loss, config)

    # Plot confusion matrix
    cm = np.array([[50, 10], [5, 35]])
    visualizer.plot_confusion_matrix(cm, ['Class 0', 'Class 1'], title='Confusion Matrix')

    # Plot distribution
    data = np.random.normal(0, 1, 1000)
    visualizer.plot_distribution(data, title='Normal Distribution')

    # Plot correlation matrix
    df = pd.DataFrame(np.random.rand(5, 5), columns=['A', 'B', 'C', 'D', 'E'])
    visualizer.plot_correlation_matrix(df, title='Correlation Matrix')

    # Plot learning curve
    train_sizes = np.linspace(0.1, 1.0, 5)
    train_scores = np.random.rand(5, 3)
    val_scores = np.random.rand(5, 3)
    visualizer.plot_learning_curve(train_scores.tolist(), val_scores.tolist(), train_sizes.tolist(), title='Learning Curve')

    # Plot feature importance
    features = ['Feature A', 'Feature B', 'Feature C', 'Feature D', 'Feature E']
    importance = np.random.rand(5)
    visualizer.plot_feature_importance(importance, features, title='Feature Importance')

    # Plot ROC curve
    fpr = np.linspace(0, 1, 100)
    tpr = np.sqrt(fpr)  # Example ROC curve
    roc_auc = 0.8
    visualizer.plot_roc_curve(fpr, tpr, roc_auc, title='ROC Curve')

    # Plot PCA
    data = np.random.rand(100, 10)
    visualizer.plot_pca(data, title='PCA Visualization')

    # Plot t-SNE
    visualizer.plot_tsne(data, title='t-SNE Visualization')

    # Save a specific plot
    visualizer.save_plot('train_loss', 'train_loss_plot.png')

    # Show all plots
    visualizer.show_plots()

    # Reset the visualizer
    visualizer.reset()

    print("All visualizations completed!")

# Additional examples for non-Plotly usage
if __name__ == "__main__":
    matplotlib_visualizer = AdvancedVisualizer(style="seaborn", use_plotly=False)

    # Example of using custom style for matplotlib
    custom_config = PlotConfig(
        title="Custom Styled Plot",
        xlabel="X-axis",
        ylabel="Y-axis",
        plot_type="scatter",
        color="green",
        marker="s",
        custom_style={"facecolor": "lightgray"}
    )
    matplotlib_visualizer.plot_metric("custom_plot", np.random.rand(20).tolist(), custom_config)

    # Example of multiple metrics on the same plot
    multi_config = PlotConfig(title="Multiple Metrics", xlabel="Time", ylabel="Value")
    matplotlib_visualizer.plot_metric("metric1", np.random.rand(10).tolist(), multi_config)
    multi_config.color = "red"
    matplotlib_visualizer.plot_metric("metric2", np.random.rand(10).tolist(), multi_config)

    matplotlib_visualizer.show_plots()

print("Advanced Visualizer demonstration completed!")