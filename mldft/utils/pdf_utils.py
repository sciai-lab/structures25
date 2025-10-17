import os
from contextlib import AbstractContextManager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import TracebackType
from typing import Type

import pypdf
from loguru import logger
from matplotlib import pyplot as plt


def add_bookmark_dict(bookmark_dict: dict, in_pdf_path: str, out_pdf_path: str = None) -> None:
    """Add bookmarks to a PDF file, as specified by a dictionary.

    Args:
        bookmark_dict: A (possibly nested) dictionary specifying the bookmarks to add to the PDF. The keys are the
            titles of the bookmarks, and the values are either integers (page numbers) or tuples of (int, dict),
            where the first element is the page number and the second element is a dictionary specifying the sub-bookmarks.
            Example:

            .. code-block::

                {
                    'Chapter 1': 1,
                    'Chapter 2': 2,
                    'Chapter 3': (3, {
                        'Section 3.1': 3,
                        'Section 3.2': 4,
                        'Section 3.3': (5, {
                            'Subsection 3.3.1': 5,
                            'Subsection 3.3.2': 6,
                        }),
                    }),
                }

        in_pdf_path: The path to the input PDF file.
        out_pdf_path: The path to the output PDF file. If not specified, the input PDF file will be overwritten.
    """
    out_pdf_path = in_pdf_path if out_pdf_path is None else out_pdf_path

    writer = pypdf.PdfWriter(clone_from=in_pdf_path)

    def add_bookmark_dict_recursive(
        sub_bookmark_dict: dict, parent: pypdf.generic.IndirectObject = None
    ):
        """Recursively add bookmarks to the PDF.

        Args:
            sub_bookmark_dict: A (possibly nested) dictionary specifying the bookmarks to add to the PDF
                (see :func:`add_bookmark_dict`).
            parent: The parent bookmark to add the bookmarks to. If not specified, the bookmarks will be added to the
                root.
        """
        for title, value in sub_bookmark_dict.items():
            if isinstance(value, int):
                page_number = value
                writer.add_outline_item(title=title, page_number=page_number, parent=parent)
            else:
                assert isinstance(
                    value, tuple
                ), f"If not int, must be tuple of (int, dict). Got {value}"
                assert isinstance(
                    value[0], int
                ), f"If not int, must be tuple of (int, dict). Got {value}"
                assert isinstance(
                    value[1], dict
                ), f"If not int, must be tuple of (int, dict). Got {value}"
                page_number, sub_dict = value
                bookmark = writer.add_outline_item(
                    title=title, page_number=page_number, parent=parent
                )
                add_bookmark_dict_recursive(sub_dict, parent=bookmark)

    add_bookmark_dict_recursive(bookmark_dict)

    writer.write(out_pdf_path)
    writer.close()


def directory_to_pdf_with_outline(
    directory: str, out_pdf_path: str = None, order: str = "time"
) -> str:
    """Convert a (possibly nested) directory containing pdfs (e.g. matplotlib plots) to a single
    PDF file, with an outline based on the directory structure.

    Args:
        directory: The path to the directory to convert to a PDF file.
        out_pdf_path: The path to the output PDF file. If not specified, the output will be saved next to the
            input directory.
        order: The order to sort the files and directories in. Can be either "time" or "name". Defaults to "time".

    Returns:
        str: The path to the output PDF file.
    """

    writer = pypdf.PdfWriter()

    if not os.path.isdir(directory):
        raise NotADirectoryError(f"{directory} is not a directory")
    directory = Path(directory)

    if out_pdf_path is None:
        out_pdf_path = directory.parent
    if os.path.isdir(out_pdf_path):
        out_pdf_path = os.path.join(out_pdf_path, directory.name + ".pdf")

    if order == "time":
        sort_key = os.path.getmtime
    else:

        def sort_key(x):
            return x.name

    def add_directory_recursive(
        sub_directory: Path, current_page: int = 0, outline_parent=None
    ) -> int:
        """Recursively add directories to the PDF.

        Args:
            sub_directory: The directory to add to the PDF.
            current_page: The current page number.
            outline_parent: The parent bookmark to add the bookmarks to. If not specified, the bookmarks will be added to the
                root.

        Returns:
            int: The current page number.
        """
        for file in sorted(sub_directory.iterdir(), key=sort_key):
            if file.is_dir():
                outline_item = writer.add_outline_item(
                    title=file.name, page_number=current_page, parent=outline_parent
                )
                current_page = add_directory_recursive(
                    file, current_page=current_page, outline_parent=outline_item
                )
            elif file.is_file():
                if file.suffix == ".pdf":
                    reader = pypdf.PdfReader(file)
                    for page in reader.pages:
                        writer.add_page(page)
                    writer.add_outline_item(
                        title=file.with_suffix("").name,
                        page_number=current_page,
                        parent=outline_parent,
                    )
                    current_page += len(reader.pages)
                else:
                    logger.warning(f"Skipping non-pdf file {file}")
        return current_page

    add_directory_recursive(directory)

    writer.write(out_pdf_path)
    writer.close()
    return out_pdf_path


class HierarchicalPlotPDF(AbstractContextManager):
    """A context manager for saving multiple plots to a PDF file, with a hierarchical outline.

    Example:
        >>> with HierarchicalPlotPDF('out.pdf') as pdf:
        ...     plt.plot([1, 2, 3])
        ...     pdf.savefig('plot1')
        ...     plt.scatter([1, 2, 3], [4, 5, 6])
        ...     pdf.savefig('scatters/scatter1')
        ...     plt.scatter([2, 3, 4], [5, 6, 7])
        ...     pdf.savefig('scatters/scatter2')

        Will save three plots to a PDF file ``out.pdf``, with the following outline:
            - plot1
            - scatters
                - scatter1
                - scatter2
    """

    def __init__(
        self, out_pdf_path: Path | str | None, individual_plot_directory: Path | str | None = None
    ) -> None:
        """Create a new HierarchicalPlotPDF context manager.

        Args:
            out_pdf_path: The path to the output PDF file. If None is specified, there no pdf with all plots will be
                compiled.
            individual_plot_directory: The path to the directory to save the individual plots to. If not specified,
                a temporary directory next to the ``out_pdf_path`` will be used.
        """
        if out_pdf_path is None and individual_plot_directory is None:
            logger.warning(
                "Both individual_plot_directory and out_pdf_path are None, nothing will be saved."
            )
        self.out_pdf_path = out_pdf_path
        self.individual_plot_directory = individual_plot_directory
        self.tempdir = None

    def __enter__(self) -> "HierarchicalPlotPDF":
        """Enter the context manager.

        Create a temporary directory if individual_plot_directory is None.
        """
        if self.individual_plot_directory is not None:
            os.makedirs(self.individual_plot_directory, exist_ok=True)
        if self.individual_plot_directory is None:
            self.tempdir = TemporaryDirectory(
                dir=os.path.dirname(self.out_pdf_path) if self.out_pdf_path is not None else None,
                prefix="individual_plots_",
            )
            self.individual_plot_directory = self.tempdir.name
        return self

    def savefig(self, path: str, **savefig_kwargs) -> None:
        """Save the current figure to a PDF file, at the specified path relative to the
        individual_plot_directory.

        Args:
            path: The path to save the figure to, relative to the individual_plot_directory.
            savefig_kwargs: Additional keyword arguments to pass to :meth:`matplotlib.pyplot.savefig`.
        """
        full_path = os.path.join(self.individual_plot_directory, path + ".pdf")
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        plt.savefig(full_path, **savefig_kwargs)

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Exit the context manager.

        Save the individual plots to a PDF file, and clean up the temporary directory.
        """
        # only save if no exception occurred (e.g. to avoid saving incomplete files on KeyboardInterrupt)
        if exc_type is None and self.out_pdf_path is not None:
            # save a single pdf with outline
            directory_to_pdf_with_outline(
                self.individual_plot_directory, out_pdf_path=self.out_pdf_path
            )
            # If there's an exception we don't clean up to not lose the individual plots
            if self.tempdir is not None:
                self.tempdir.cleanup()
