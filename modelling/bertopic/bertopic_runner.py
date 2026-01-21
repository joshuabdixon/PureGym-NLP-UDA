from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.io as pio
from bertopic import BERTopic
from umap import UMAP


@dataclass(frozen=True)
class BERTopicRunResult:
    """
    Container for BERTopic run outputs.

    Parameters
    ----------
    label
        Short label used for filenames.
    docs
        Input documents used for modelling.
    model
        Fitted BERTopic model.
    topics
        Topic assignment per document.
    probabilities
        Topic probabilities per document, if available.
    model_path
        Directory where the model was saved.
    plot_paths
        Mapping of artefact name to saved path (plots and tables).
    topic_info
        DataFrame returned by ``get_topic_info``.
    top_topics_table
        Summary table of top topics by frequency with top words.
    """
    label: str
    docs: list[str]
    model: BERTopic
    topics: list[int]
    probabilities: Any
    model_path: Path
    plot_paths: dict[str, Path]
    topic_info: pd.DataFrame
    top_topics_table: pd.DataFrame


class BERTopicRunner:
    """
    Fit BERTopic on a dataset, save the model, and generate standard outputs.

    Parameters
    ----------
    model_dir
        Directory where BERTopic models will be saved.
    plot_dir
        Directory where interactive Plotly outputs will be saved as HTML.
    table_dir
        Directory where CSV tables will be saved.
    seed
        Random seed for reproducibility (applied to UMAP).
    top_n_topics
        Number of top topics (by frequency) used for summaries and focused plots.
    n_words_barchart
        Number of words per topic in the barchart.
    min_topic_size
        Minimum size of a topic. Smaller values generally produce more topics
        (useful for smaller subsets), but can increase noise.
    umap_n_neighbors
        UMAP n_neighbors parameter.
    umap_n_components
        UMAP n_components parameter.
    umap_min_dist
        UMAP min_dist parameter.
    umap_metric
        UMAP distance metric.
    show_plots
        If True, attempt to render Plotly figures inside the notebook.
        If False, figures are saved to HTML only (recommended for environments
        missing nbformat).
    save_png
        If True, also export PNG versions of supported Plotly figures.
        Requires the Plotly image export dependency (kaleido).
    png_scale
        Scale factor for PNG export. Higher values increase resolution.
    """

    def __init__(
        self,
        *,
        model_dir: Path,
        plot_dir: Path,
        table_dir: Path,
        seed: int,
        top_n_topics: int = 5,
        n_words_barchart: int = 5,
        min_topic_size: int = 10,
        umap_n_neighbors: int = 15,
        umap_n_components: int = 5,
        umap_min_dist: float = 0.0,
        umap_metric: str = "cosine",
        show_plots: bool = False,
        save_png: bool = False,
        png_scale: int = 2,
    ) -> None:
        self._model_dir = Path(model_dir)
        self._plot_dir = Path(plot_dir)
        self._table_dir = Path(table_dir)
        self._seed = int(seed)

        self._top_n_topics = int(top_n_topics)
        self._n_words_barchart = int(n_words_barchart)
        self._min_topic_size = int(min_topic_size)

        self._umap_n_neighbors = int(umap_n_neighbors)
        self._umap_n_components = int(umap_n_components)
        self._umap_min_dist = float(umap_min_dist)
        self._umap_metric = str(umap_metric)

        self._show_plots = bool(show_plots)
        self._save_png = bool(save_png)
        self._png_scale = int(png_scale)

        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._plot_dir.mkdir(parents=True, exist_ok=True)
        self._table_dir.mkdir(parents=True, exist_ok=True)

        np.random.seed(self._seed)

    @property
    def model_dir(self) -> Path:
        """
        Return the configured model directory.

        Returns
        -------
        pathlib.Path
            Model directory.
        """
        return self._model_dir

    @property
    def plot_dir(self) -> Path:
        """
        Return the configured plot directory.

        Returns
        -------
        pathlib.Path
            Plot directory.
        """
        return self._plot_dir

    @property
    def table_dir(self) -> Path:
        """
        Return the configured table directory.

        Returns
        -------
        pathlib.Path
            Table directory.
        """
        return self._table_dir

    def run(
        self,
        df: pd.DataFrame,
        *,
        label: str,
        text_col: str,
        verbose: bool = True,
    ) -> BERTopicRunResult:
        """
        Fit BERTopic, save model, and generate standard plots and tables.

        Parameters
        ----------
        df
            Input dataset.
        label
            Short label used for filenames (e.g. "non_negative", "negative").
        text_col
            Name of the column containing the review text.
        verbose
            If True, BERTopic prints progress logs.

        Returns
        -------
        BERTopicRunResult
            Outputs including model, topic info, and saved artefact paths.
        """
        docs = self._to_documents(df, text_col=text_col)
        if not docs:
            raise ValueError(f"No documents found for label='{label}'. Check '{text_col}' and filtering.")

        umap_model = UMAP(
            n_neighbors=self._umap_n_neighbors,
            n_components=self._umap_n_components,
            min_dist=self._umap_min_dist,
            metric=self._umap_metric,
            random_state=self._seed,
        )

        topic_model = BERTopic(
            umap_model=umap_model,
            min_topic_size=self._min_topic_size,
            verbose=verbose,
        )

        topics, probabilities = topic_model.fit_transform(docs)

        model_path = self._model_dir / f"bertopic_{label}"
        topic_model.save(model_path)

        topic_info = topic_model.get_topic_info()
        top_topics_table = self._top_topics_word_table(
            topic_model,
            top_n_topics=self._top_n_topics,
            top_n_words=10,
        )

        plot_paths = self._save_standard_plots(topic_model, label=label)
        table_paths = self._save_standard_tables(
            label=label,
            topic_info=topic_info,
            top_topics_table=top_topics_table,
        )

        artefact_paths = {**plot_paths, **table_paths}

        return BERTopicRunResult(
            label=label,
            docs=docs,
            model=topic_model,
            topics=topics,
            probabilities=probabilities,
            model_path=model_path,
            plot_paths=artefact_paths,
            topic_info=topic_info,
            top_topics_table=top_topics_table,
        )

    def _to_documents(self, df: pd.DataFrame, *, text_col: str) -> list[str]:
        """
        Convert a DataFrame text column to a list of documents.

        Parameters
        ----------
        df
            Input dataset.
        text_col
            Text column name.

        Returns
        -------
        list[str]
            List of documents.
        """
        if text_col not in df.columns:
            raise KeyError(f"'{text_col}' not found in DataFrame")

        return (
            df[text_col]
            .dropna()
            .astype(str)
            .str.strip()
            .loc[lambda s: s.ne("")]
            .tolist()
        )

    def _save_plotly_html(self, fig, filename: str) -> Path:
        """
        Save a Plotly figure as an HTML file in ``plot_dir``.

        Parameters
        ----------
        fig
            Plotly figure.
        filename
            Output filename.

        Returns
        -------
        pathlib.Path
            Saved HTML path.
        """
        save_path = self._plot_dir / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        pio.write_html(fig, file=str(save_path), auto_open=False, include_plotlyjs="cdn")
        return save_path

    def _save_plotly_png(self, fig, filename: str) -> Path:
        """
        Save a Plotly figure as a PNG file in ``plot_dir``.

        Notes
        -----
        PNG export requires the Plotly image export dependency (kaleido). If it
        is not installed, this method raises a RuntimeError.

        Parameters
        ----------
        fig
            Plotly figure.
        filename
            Output filename.

        Returns
        -------
        pathlib.Path
            Saved PNG path.
        """
        save_path = self._plot_dir / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            fig.write_image(str(save_path), scale=self._png_scale)
        except ValueError as e:
            raise RuntimeError(
                "PNG export failed. Install kaleido (e.g. `pip install -U kaleido`) "
                "or set save_png=False."
            ) from e

        return save_path

    def _maybe_show(self, fig) -> None:
        """
        Try to show a Plotly figure if enabled.

        Parameters
        ----------
        fig
            Plotly figure.
        """
        if not self._show_plots:
            return

        try:
            fig.show()
        except ValueError:
            return

    def _save_standard_plots(self, model: BERTopic, *, label: str) -> dict[str, Path]:
        """
        Generate and save standard BERTopic plots.

        Parameters
        ----------
        model
            Fitted BERTopic model.
        label
            Label used in filenames.

        Returns
        -------
        dict[str, pathlib.Path]
            Saved plot paths.
        """
        plot_paths: dict[str, Path] = {}

        fig_topics = model.visualize_topics()
        self._maybe_show(fig_topics)
        plot_paths["intertopic_distance_html"] = self._save_plotly_html(
            fig_topics,
            f"bertopic_{label}_intertopic_distance.html",
        )
        if self._save_png:
            plot_paths["intertopic_distance_png"] = self._save_plotly_png(
                fig_topics,
                f"bertopic_{label}_intertopic_distance.png",
            )

        fig_bar = model.visualize_barchart(
            top_n_topics=self._top_n_topics,
            n_words=self._n_words_barchart,
        )
        self._maybe_show(fig_bar)
        plot_paths["barchart_top_topics_html"] = self._save_plotly_html(
            fig_bar,
            f"bertopic_{label}_barchart_top{self._top_n_topics}.html",
        )
        if self._save_png:
            plot_paths["barchart_top_topics_png"] = self._save_plotly_png(
                fig_bar,
                f"bertopic_{label}_barchart_top{self._top_n_topics}.png",
            )

        fig_heat_all = model.visualize_heatmap(
            title=f"Similarity Matrix (All Topics) - {label}",
        )
        self._maybe_show(fig_heat_all)
        plot_paths["heatmap_all_topics_html"] = self._save_plotly_html(
            fig_heat_all,
            f"bertopic_{label}_heatmap_all_topics.html",
        )
        if self._save_png:
            plot_paths["heatmap_all_topics_png"] = self._save_plotly_png(
                fig_heat_all,
                f"bertopic_{label}_heatmap_all_topics.png",
            )

        fig_heat_top = model.visualize_heatmap(
            top_n_topics=self._top_n_topics,
            title=f"Similarity Matrix (Top {self._top_n_topics} Topics) - {label}",
        )
        self._maybe_show(fig_heat_top)
        plot_paths["heatmap_top_topics_html"] = self._save_plotly_html(
            fig_heat_top,
            f"bertopic_{label}_heatmap_top{self._top_n_topics}.html",
        )
        if self._save_png:
            plot_paths["heatmap_top_topics_png"] = self._save_plotly_png(
                fig_heat_top,
                f"bertopic_{label}_heatmap_top{self._top_n_topics}.png",
            )

        return plot_paths

    def _save_standard_tables(
        self,
        *,
        label: str,
        topic_info: pd.DataFrame,
        top_topics_table: pd.DataFrame,
        index: bool = False,
    ) -> dict[str, Path]:
        """
        Save standard BERTopic tables to CSV.

        Parameters
        ----------
        label
            Label used in filenames.
        topic_info
            Output from ``model.get_topic_info()``.
        top_topics_table
            Summary table of top topics with top words.
        index
            If True, include DataFrame index in CSVs.

        Returns
        -------
        dict[str, pathlib.Path]
            Saved table paths.
        """
        self._table_dir.mkdir(parents=True, exist_ok=True)

        topic_info_path = self._table_dir / f"bertopic_{label}_topic_info.csv"
        topic_info.to_csv(topic_info_path, index=index)

        top_topics_path = self._table_dir / f"bertopic_{label}_top_topics.csv"
        top_topics_table.to_csv(top_topics_path, index=index)

        return {
            "topic_info_csv": topic_info_path,
            "top_topics_csv": top_topics_path,
        }

    def _top_topics_word_table(
        self,
        model: BERTopic,
        *,
        top_n_topics: int,
        top_n_words: int,
    ) -> pd.DataFrame:
        """
        Create a table of top topics by frequency with their top words.

        Parameters
        ----------
        model
            Fitted BERTopic model.
        top_n_topics
            Number of topics to include (excluding outlier topic -1).
        top_n_words
            Number of words per topic.

        Returns
        -------
        pandas.DataFrame
            Summary table.
        """
        topic_info = model.get_topic_info()
        topic_info_no_outliers = topic_info[topic_info["Topic"] != -1].copy()
        top_topics = topic_info_no_outliers.head(top_n_topics)

        rows: list[dict[str, Any]] = []
        for topic_id in top_topics["Topic"].tolist():
            words_scores = (model.get_topic(topic_id) or [])[:top_n_words]
            rows.append(
                {
                    "Topic": int(topic_id),
                    "Count": int(
                        topic_info_no_outliers.loc[
                            topic_info_no_outliers["Topic"] == topic_id, "Count"
                        ].iloc[0]
                    ),
                    "TopWords": ", ".join([w for w, _ in words_scores]),
                }
            )

        return pd.DataFrame(rows).sort_values("Count", ascending=False).reset_index(drop=True)
