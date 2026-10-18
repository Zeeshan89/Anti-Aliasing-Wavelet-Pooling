import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors


# ============================================================
# USER CONFIGURATION
# ============================================================

CORE = {
    "title": "Computer Science",
    "color": "#123B5D",
    "radius": 0.58,

    "title_color": "black",
    "title_alpha": 0.90,
    "title_size": 20,

    # Position of the curved title within the core.
    "title_radius_position": 0.68,
}


RINGS = [
    {
        "title": "Software Engineering",

        # Number of slices in this ring.
        "n_slices": 3,

        # None creates equal-sized slices.
        # Otherwise, provide one value per slice.
        "values": None,

        # Exact labels. No numbers, bullets, or percentages are added.
        "labels": [
            "Frontend Development",
            "Backend Development",
            "Database",
        ],

        # None means use the automatic ring color.
        # Replace any None with a custom color for that slice.
        "slice_colors": [
            None,
            None, #"#C45A3A",
            None,
        ],

        # Staggers this ring relative to the others.
        "start_angle": 90,

        "width": 0.48,
        "gap": 0.02,

        # Curved slice-label settings.
        "label_size": 13,
        "label_color": "white",
        "label_radius_position": 0.50,

        # Curved ring-title settings.
        "title_size": 20,
        "title_color": "black",
        "title_alpha": 0.90,
        "title_radius_position": 0.74,
    },

    {
        "title": "AI and Data Science",

        "n_slices": 3,
        "values": None,

        "labels": [
            "Pattern Recognition",
            "Clustering",
            "Prediction",
        ],

        "slice_colors": [
            None,
            None,
            None,
        ],

        "start_angle": 50,

        "width": 0.48,
        "gap": 0.02,

        "label_size": 13,
        "label_color": "white",
        "label_radius_position": 0.50,

        "title_size": 20,
        "title_color": "black",
        "title_alpha": 0.90,
        "title_radius_position": 0.74,
    },

    {
        "title": "Physical Domain Science",

        "n_slices": 3,
        "values": None,

        "labels": [
            "Characterize Climate Regimes Across Global Cities",
            "Predict Environmental Forcing Influencing South Asian Monsoon",
            "Identify Dominant Precipitation Types",
        ],

        # Highlight this slice in light orange/red to show
        # this is work we still have to start.
        "slice_colors": [
            None,
            "#F4A582",
            None,
        ],

        "start_angle": 10,

        "width": 0.48,
        "gap": 0.02,

        "label_size": 13,
        "label_color": "white",
        "label_radius_position": 0.50,

        "title_size": 18,
        "title_color": "black",
        "title_alpha": 0.90,
        "title_radius_position": 0.74,
    },
]


# ============================================================
# AUTOMATIC COLOR THEME
# ============================================================

# All rings use this color family.
BASE_COLOR = "#174A70"

# Controls how light the outermost ring becomes.
# 0.0 means no lightening.
# 1.0 means white.
MAX_RING_LIGHTENING = 0.68


# ============================================================
# CHART SETTINGS
# ============================================================

DEFAULT_START_ANGLE = 90
CLOCKWISE = True

EDGE_COLOR = "white"
EDGE_WIDTH = 1.8

FIGURE_SIZE = (13, 13)
FIGURE_TITLE = ""

SHOW_PLOT = True


# ============================================================
# OUTPUT SETTINGS
# ============================================================

OUTPUT_BASENAME = "staggered_cascading_pie"

SAVE_PNG = True
SAVE_PDF = True

PNG_DPI = 600


# ============================================================
# COLOR FUNCTIONS
# ============================================================

def blend_with_white(color, amount):
    """
    Lighten a color by blending it with white.

    amount=0 returns the original color.
    amount=1 returns white.
    """
    rgb = np.asarray(
        mcolors.to_rgb(color),
        dtype=float,
    )

    white = np.ones(3)

    return tuple(
        rgb * (1.0 - amount)
        + white * amount
    )


def get_automatic_ring_color(
    ring_index,
    number_of_rings,
):
    """
    Return one uniform color for an entire ring.

    Colors become progressively lighter from the inner ring
    to the outer ring.
    """
    if number_of_rings <= 1:
        lightening = 0.0
    else:
        lightening = (
            MAX_RING_LIGHTENING
            * ring_index
            / (number_of_rings - 1)
        )

    return blend_with_white(
        BASE_COLOR,
        lightening,
    )


def resolve_slice_colors(
    automatic_ring_color,
    slice_color_overrides,
    number_of_slices,
):
    """
    Use the automatic ring color for every slice unless a
    specific slice has a custom color override.
    """
    if slice_color_overrides is None:
        return [
            automatic_ring_color
        ] * number_of_slices

    if len(slice_color_overrides) != number_of_slices:
        raise ValueError(
            "slice_colors must contain one entry per slice."
        )

    resolved_colors = []

    for override in slice_color_overrides:
        if override is None:
            resolved_colors.append(
                automatic_ring_color
            )
        else:
            if not mcolors.is_color_like(override):
                raise ValueError(
                    f"Invalid slice color: {override}"
                )

            resolved_colors.append(
                override
            )

    return resolved_colors


# ============================================================
# VALIDATION
# ============================================================

def validate_settings():
    if not mcolors.is_color_like(CORE["color"]):
        raise ValueError(
            f"Invalid core color: {CORE['color']}"
        )

    if not mcolors.is_color_like(CORE["title_color"]):
        raise ValueError(
            f"Invalid core title color: {CORE['title_color']}"
        )

    if not RINGS:
        raise ValueError(
            "At least one ring must be defined."
        )

    for ring_number, ring in enumerate(RINGS, start=1):
        number_of_slices = ring["n_slices"]
        labels = ring["labels"]
        values = ring.get("values")
        slice_colors = ring.get("slice_colors")

        if number_of_slices < 1:
            raise ValueError(
                f"Ring {ring_number} must have at least one slice."
            )

        if len(labels) != number_of_slices:
            raise ValueError(
                f"Ring {ring_number} has "
                f"{number_of_slices} slices but "
                f"{len(labels)} labels."
            )

        if slice_colors is not None:
            if len(slice_colors) != number_of_slices:
                raise ValueError(
                    f"Ring {ring_number} has "
                    f"{number_of_slices} slices but "
                    f"{len(slice_colors)} color entries."
                )

        if values is not None:
            if len(values) != number_of_slices:
                raise ValueError(
                    f"Ring {ring_number} has "
                    f"{number_of_slices} slices but "
                    f"{len(values)} values."
                )

            if any(value < 0 for value in values):
                raise ValueError(
                    f"Ring {ring_number} contains a negative value."
                )

            if sum(values) <= 0:
                raise ValueError(
                    f"Ring {ring_number} values must sum "
                    "to more than zero."
                )

        label_color = ring.get(
            "label_color",
            "white",
        )

        title_color = ring.get(
            "title_color",
            "white",
        )

        if not mcolors.is_color_like(label_color):
            raise ValueError(
                f"Invalid label color in ring "
                f"{ring_number}: {label_color}"
            )

        if not mcolors.is_color_like(title_color):
            raise ValueError(
                f"Invalid title color in ring "
                f"{ring_number}: {title_color}"
            )


# ============================================================
# CURVED TEXT FUNCTIONS
# ============================================================

def character_width_weights(text):
    """
    Estimate relative character widths for more natural spacing
    along a circular arc.
    """
    narrow_characters = set("ilI.,:'|!")
    wide_characters = set("MWmw")

    weights = []

    for character in text:
        if character == " ":
            weight = 0.50
        elif character in narrow_characters:
            weight = 0.55
        elif character in wide_characters:
            weight = 1.30
        else:
            weight = 1.0

        weights.append(weight)

    return np.asarray(
        weights,
        dtype=float,
    )


def add_text_along_arc(
    axis,
    text,
    radius,
    start_angle,
    end_angle,
    fontsize,
    color,
    alpha=1.0,
    fontweight="normal",
    face_down=False,
):
    """
    Draw text along an arc by positioning each character
    separately.
    """
    if not text:
        return

    characters = list(text)
    weights = character_width_weights(text)

    character_centers = (
        np.cumsum(weights)
        - weights / 2.0
    ) / weights.sum()

    angles = (
        start_angle
        + character_centers
        * (end_angle - start_angle)
    )

    for character, angle_degrees in zip(
        characters,
        angles,
    ):
        if character == " ":
            continue

        angle_radians = np.radians(
            angle_degrees
        )

        x_position = (
            radius
            * np.cos(angle_radians)
        )

        y_position = (
            radius
            * np.sin(angle_radians)
        )

        if face_down:
            # Curved text on the lower half, with the letters
            # facing downward.
            rotation = angle_degrees + 90.0
        else:
            rotation = angle_degrees - 90.0

            normalized_angle = (
                angle_degrees % 360
            )

            if 90 < normalized_angle < 270:
                rotation += 180.0

        axis.text(
            x_position,
            y_position,
            character,
            fontsize=fontsize,
            color=color,
            alpha=alpha,
            fontweight=fontweight,
            ha="center",
            va="center",
            rotation=rotation,
            rotation_mode="anchor",
            clip_on=False,
        )


def estimate_natural_arc_span(
    text,
    fontsize,
    radius,
    spacing_factor=0.0037,
):
    """
    Estimate a natural arc width for a title.

    The text is centered at 6 o'clock but is not stretched
    from 9 o'clock to 3 o'clock.
    """
    if not text:
        return 0.0

    weights = character_width_weights(text)

    estimated_text_length = (
        weights.sum()
        * fontsize
        * spacing_factor
    )

    span_radians = (
        estimated_text_length
        / max(radius, 0.01)
    )

    span_degrees = np.degrees(
        span_radians
    )

    return float(
        np.clip(
            span_degrees,
            12.0,
            170.0,
        )
    )


def add_bottom_centered_curved_title(
    axis,
    text,
    radius,
    fontsize,
    color,
    alpha,
    spacing_factor=0.0037,
):
    """
    Draw naturally spaced curved text centered at 6 o'clock.

    The text follows the lower circular arc and faces downward.
    It occupies only the angular width needed for the text.
    """
    if not text:
        return

    arc_span = estimate_natural_arc_span(
        text=text,
        fontsize=fontsize,
        radius=radius,
        spacing_factor=spacing_factor,
    )

    center_angle = 270.0

    start_angle = (
        center_angle
        - arc_span / 2.0
    )

    end_angle = (
        center_angle
        + arc_span / 2.0
    )

    add_text_along_arc(
        axis=axis,
        text=text,
        radius=radius,
        start_angle=start_angle,
        end_angle=end_angle,
        fontsize=fontsize,
        color=color,
        alpha=alpha,
        fontweight="bold",
        face_down=True,
    )


def add_curved_slice_label(
    axis,
    text,
    theta1,
    theta2,
    radius,
    fontsize,
    color,
):
    """
    Draw naturally spaced curved text centered within a slice.

    The full word keeps one consistent reading direction. Labels on
    the upper half follow the arc from left to right; labels on the
    lower half follow the lower arc from left to right. No part of a
    word is reflected or reversed.
    """
    if not text:
        return

    slice_span = abs(theta2 - theta1)

    if slice_span < 4:
        return

    midpoint = (theta1 + theta2) / 2.0
    midpoint_normalized = midpoint % 360

    natural_span = estimate_natural_arc_span(
        text=text,
        fontsize=fontsize,
        radius=radius,
        spacing_factor=0.0034,
    )

    # Keep the label compact and centered within the slice.
    label_span = min(natural_span, slice_span * 0.55)

    characters = list(text)
    weights = character_width_weights(text)
    character_centers = (
        np.cumsum(weights) - weights / 2.0
    ) / weights.sum()

    if 0 <= midpoint_normalized <= 180:
        # Upper half: first character starts on the left side of the
        # label and the word proceeds toward the right.
        start_angle = midpoint + label_span / 2.0
        end_angle = midpoint - label_span / 2.0
        rotations = lambda angle: angle - 90.0
    else:
        # Lower half: first character starts on the left side of the
        # label and the word proceeds toward the right.
        start_angle = midpoint - label_span / 2.0
        end_angle = midpoint + label_span / 2.0
        rotations = lambda angle: angle + 90.0

    angles = (
        start_angle
        + character_centers * (end_angle - start_angle)
    )

    for character, angle_degrees in zip(characters, angles):
        if character == " ":
            continue

        angle_radians = np.radians(angle_degrees)
        x_position = radius * np.cos(angle_radians)
        y_position = radius * np.sin(angle_radians)

        axis.text(
            x_position,
            y_position,
            character,
            fontsize=fontsize,
            color=color,
            alpha=1.0,
            fontweight="bold",
            ha="center",
            va="center",
            rotation=rotations(angle_degrees),
            rotation_mode="anchor",
            clip_on=False,
        )


# ============================================================
# CHART CREATION
# ============================================================

def create_chart():
    validate_settings()

    figure, axis = plt.subplots(
        figsize=FIGURE_SIZE
    )

    current_outer_radius = CORE["radius"]
    number_of_rings = len(RINGS)

    for ring_index, ring in enumerate(RINGS):
        number_of_slices = ring["n_slices"]
        values = ring.get("values")

        # Equal-sized slices by default.
        if values is None:
            values = np.ones(
                number_of_slices
            )

        ring_width = ring.get(
            "width",
            0.48,
        )

        ring_gap = ring.get(
            "gap",
            0.02,
        )

        ring_inner_radius = (
            current_outer_radius
            + ring_gap
        )

        ring_outer_radius = (
            ring_inner_radius
            + ring_width
        )

        automatic_ring_color = (
            get_automatic_ring_color(
                ring_index,
                number_of_rings,
            )
        )

        slice_colors = resolve_slice_colors(
            automatic_ring_color,
            ring.get("slice_colors"),
            number_of_slices,
        )

        ring_start_angle = ring.get(
            "start_angle",
            DEFAULT_START_ANGLE,
        )

        wedges, _ = axis.pie(
            values,
            radius=ring_outer_radius,
            colors=slice_colors,
            startangle=ring_start_angle,
            counterclock=not CLOCKWISE,
            wedgeprops={
                "width": ring_width,
                "edgecolor": EDGE_COLOR,
                "linewidth": EDGE_WIDTH,
            },
        )

        label_radius_position = ring.get(
            "label_radius_position",
            0.50,
        )

        label_radius = (
            ring_inner_radius
            + ring_width
            * label_radius_position
        )

        for wedge, label in zip(
            wedges,
            ring["labels"],
        ):
            add_curved_slice_label(
                axis=axis,
                text=label,
                theta1=wedge.theta1,
                theta2=wedge.theta2,
                radius=label_radius,
                fontsize=ring.get(
                    "label_size",
                    9,
                ),
                color=ring.get(
                    "label_color",
                    "white",
                ),
            )

        title_radius_position = ring.get(
            "title_radius_position",
            0.74,
        )

        title_radius = (
            ring_inner_radius
            + ring_width
            * title_radius_position
        )

        add_bottom_centered_curved_title(
            axis=axis,
            text=ring["title"],
            radius=title_radius,
            fontsize=ring.get(
                "title_size",
                18,
            ),
            color=ring.get(
                "title_color",
                "white",
            ),
            alpha=ring.get(
                "title_alpha",
                0.38,
            ),
        )

        current_outer_radius = (
            ring_outer_radius
        )

    # Draw the central core.
    core_circle = plt.Circle(
        (0, 0),
        CORE["radius"],
        facecolor=CORE["color"],
        edgecolor=EDGE_COLOR,
        linewidth=EDGE_WIDTH,
    )

    axis.add_artist(
        core_circle
    )

    core_title_radius = (
        CORE["radius"]
        * CORE.get(
            "title_radius_position",
            0.68,
        )
    )

    add_bottom_centered_curved_title(
        axis=axis,
        text=CORE["title"],
        radius=core_title_radius,
        fontsize=CORE["title_size"],
        color=CORE["title_color"],
        alpha=CORE["title_alpha"],
    )

    if FIGURE_TITLE:
        axis.set_title(
            FIGURE_TITLE,
            fontsize=18,
            pad=20,
        )

    axis.set_aspect("equal")
    axis.axis("off")

    plt.tight_layout()

    if SAVE_PNG:
        png_file = (
            f"{OUTPUT_BASENAME}.png"
        )

        figure.savefig(
            png_file,
            dpi=PNG_DPI,
            bbox_inches="tight",
            facecolor="white",
        )

        print(
            f"Saved: {png_file}"
        )

    if SAVE_PDF:
        pdf_file = (
            f"{OUTPUT_BASENAME}.pdf"
        )

        figure.savefig(
            pdf_file,
            bbox_inches="tight",
            facecolor="white",
        )

        print(
            f"Saved: {pdf_file}"
        )

    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    create_chart()
