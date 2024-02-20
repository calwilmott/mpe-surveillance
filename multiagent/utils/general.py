def get_line_bresenham(start, end):
    """Generate the points of a line using Bresenham's algorithm, including neighboring squares for continuous lines."""
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1

    is_steep = abs(dy) > abs(dx)  # Determine how steep the line is

    # Rotate line if steep
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # Swap start and end points if necessary
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    dx = x2 - x1
    dy = y2 - y1

    error = dx / 2.0
    ystep = 1 if y1 < y2 else -1

    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        if is_steep:
            points.append((y + ystep, x))  # Add neighboring square for steep lines
        else:
            points.append((x, y + ystep))  # Add neighboring square for shallow lines
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx

    if swapped:
        points.reverse()

    return set(points)  # Remove duplicates


def get_grid_coord(pos, grid_resolution):
    """Converts a position to grid coordinates."""
    return min(int((pos + 1) / 2 * grid_resolution), grid_resolution - 1)